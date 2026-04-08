"""
Full market simulator for strategy testing.

Multi-asset correlated GBM, Heston stochastic volatility, Merton jump-diffusion,
regime-switching, order book simulation, market impact, informed/noise traders,
flash crashes, earnings announcements, intraday patterns, correlation regime shifts,
and central bank interventions.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# 1. Multi-Asset Correlated GBM
# ---------------------------------------------------------------------------

class CorrelatedGBM:
    """Multi-asset geometric Brownian motion with Cholesky decomposition."""

    def __init__(self, n_assets: int, mus: np.ndarray, sigmas: np.ndarray,
                 corr: np.ndarray, dt: float = 1 / 252,
                 rng: Optional[np.random.Generator] = None):
        self.n_assets = n_assets
        self.mus = mus
        self.sigmas = sigmas
        self.corr = corr
        self.dt = dt
        self.rng = rng or np.random.default_rng(42)
        self.L = np.linalg.cholesky(corr + np.eye(n_assets) * 1e-8)

    def simulate(self, S0: np.ndarray, n_steps: int) -> np.ndarray:
        """Returns (n_steps+1, n_assets) price paths."""
        paths = np.zeros((n_steps + 1, self.n_assets))
        paths[0] = S0
        for t in range(n_steps):
            z = self.rng.standard_normal(self.n_assets)
            corr_z = self.L @ z
            drift = (self.mus - 0.5 * self.sigmas ** 2) * self.dt
            diffusion = self.sigmas * np.sqrt(self.dt) * corr_z
            paths[t + 1] = paths[t] * np.exp(drift + diffusion)
        return paths

    def simulate_returns(self, n_steps: int) -> np.ndarray:
        S0 = np.ones(self.n_assets) * 100
        paths = self.simulate(S0, n_steps)
        return np.diff(np.log(paths), axis=0)


# ---------------------------------------------------------------------------
# 2. Heston Stochastic Volatility
# ---------------------------------------------------------------------------

class HestonModel:
    """Heston stochastic volatility model per asset."""

    def __init__(self, kappa: float = 2.0, theta: float = 0.04,
                 xi: float = 0.3, rho: float = -0.7,
                 v0: float = 0.04, mu: float = 0.05,
                 dt: float = 1 / 252, rng: Optional[np.random.Generator] = None):
        self.kappa = kappa    # mean reversion speed
        self.theta = theta    # long-run variance
        self.xi = xi          # vol of vol
        self.rho = rho        # correlation between price and vol
        self.v0 = v0          # initial variance
        self.mu = mu
        self.dt = dt
        self.rng = rng or np.random.default_rng(42)

    def simulate(self, S0: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (prices, variances) arrays of length n_steps+1."""
        prices = np.zeros(n_steps + 1)
        variances = np.zeros(n_steps + 1)
        prices[0] = S0
        variances[0] = self.v0
        for t in range(n_steps):
            z1 = self.rng.standard_normal()
            z2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * self.rng.standard_normal()
            v = max(variances[t], 0)
            sqrt_v = np.sqrt(v)
            # Variance process (truncated)
            dv = self.kappa * (self.theta - v) * self.dt + self.xi * sqrt_v * np.sqrt(self.dt) * z2
            variances[t + 1] = max(v + dv, 1e-8)
            # Price process
            dp = (self.mu - 0.5 * v) * self.dt + sqrt_v * np.sqrt(self.dt) * z1
            prices[t + 1] = prices[t] * np.exp(dp)
        return prices, variances


class MultiAssetHeston:
    """Multi-asset Heston with correlation structure."""

    def __init__(self, n_assets: int, params: List[Dict[str, float]],
                 corr: np.ndarray, dt: float = 1 / 252,
                 rng: Optional[np.random.Generator] = None):
        self.n_assets = n_assets
        self.models = []
        self.rng = rng or np.random.default_rng(42)
        for p in params:
            self.models.append(HestonModel(**p, dt=dt, rng=self.rng))
        self.corr = corr
        self.L = np.linalg.cholesky(corr + np.eye(n_assets) * 1e-8)

    def simulate(self, S0: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        prices = np.zeros((n_steps + 1, self.n_assets))
        vols = np.zeros((n_steps + 1, self.n_assets))
        prices[0] = S0
        for i in range(self.n_assets):
            vols[0, i] = self.models[i].v0
        for t in range(n_steps):
            z = self.rng.standard_normal(self.n_assets)
            corr_z = self.L @ z
            for i in range(self.n_assets):
                m = self.models[i]
                v = max(vols[t, i], 0)
                sqrt_v = np.sqrt(v)
                z1 = corr_z[i]
                z2 = m.rho * z1 + np.sqrt(1 - m.rho ** 2) * self.rng.standard_normal()
                dv = m.kappa * (m.theta - v) * m.dt + m.xi * sqrt_v * np.sqrt(m.dt) * z2
                vols[t + 1, i] = max(v + dv, 1e-8)
                dp = (m.mu - 0.5 * v) * m.dt + sqrt_v * np.sqrt(m.dt) * z1
                prices[t + 1, i] = prices[t, i] * np.exp(dp)
        return prices, vols


# ---------------------------------------------------------------------------
# 3. Jump-Diffusion (Merton) Overlay
# ---------------------------------------------------------------------------

class MertonJumpDiffusion:
    """Merton jump-diffusion: GBM + compound Poisson jumps."""

    def __init__(self, mu: float = 0.05, sigma: float = 0.2,
                 jump_intensity: float = 2.0,
                 jump_mean: float = -0.02, jump_std: float = 0.05,
                 dt: float = 1 / 252, rng: Optional[np.random.Generator] = None):
        self.mu = mu
        self.sigma = sigma
        self.lam = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.dt = dt
        self.rng = rng or np.random.default_rng(42)

    def simulate(self, S0: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (prices, jump_indicators)."""
        prices = np.zeros(n_steps + 1)
        jumps = np.zeros(n_steps + 1)
        prices[0] = S0
        for t in range(n_steps):
            z = self.rng.standard_normal()
            n_jumps = self.rng.poisson(self.lam * self.dt)
            jump_size = 0.0
            if n_jumps > 0:
                jump_size = np.sum(self.rng.normal(self.jump_mean, self.jump_std, n_jumps))
                jumps[t + 1] = 1
            drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
            diffusion = self.sigma * np.sqrt(self.dt) * z
            prices[t + 1] = prices[t] * np.exp(drift + diffusion + jump_size)
        return prices, jumps

    def simulate_multi(self, S0: np.ndarray, n_steps: int,
                       corr: np.ndarray) -> np.ndarray:
        n = len(S0)
        L = np.linalg.cholesky(corr + np.eye(n) * 1e-8)
        prices = np.zeros((n_steps + 1, n))
        prices[0] = S0
        for t in range(n_steps):
            z = L @ self.rng.standard_normal(n)
            for i in range(n):
                n_j = self.rng.poisson(self.lam * self.dt)
                js = np.sum(self.rng.normal(self.jump_mean, self.jump_std, n_j)) if n_j > 0 else 0
                drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
                prices[t + 1, i] = prices[t, i] * np.exp(drift + self.sigma * np.sqrt(self.dt) * z[i] + js)
        return prices


# ---------------------------------------------------------------------------
# 4. Regime-Switching Dynamics
# ---------------------------------------------------------------------------

class RegimeSwitchingSimulator:
    """Market dynamics with regime-switching parameters."""

    def __init__(self, regime_params: List[Dict[str, float]],
                 transition_matrix: np.ndarray,
                 dt: float = 1 / 252, rng: Optional[np.random.Generator] = None):
        """
        regime_params: [{"mu": ..., "sigma": ..., "jump_intensity": ...}, ...]
        transition_matrix: (n_regimes, n_regimes) Markov chain
        """
        self.params = regime_params
        self.P = transition_matrix
        self.n_regimes = len(regime_params)
        self.dt = dt
        self.rng = rng or np.random.default_rng(42)

    def simulate(self, S0: float, n_steps: int,
                 initial_regime: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (prices, regime_labels)."""
        prices = np.zeros(n_steps + 1)
        regimes = np.zeros(n_steps + 1, dtype=int)
        prices[0] = S0
        regimes[0] = initial_regime
        regime = initial_regime
        for t in range(n_steps):
            # Regime transition
            regime = self.rng.choice(self.n_regimes, p=self.P[regime])
            regimes[t + 1] = regime
            p = self.params[regime]
            mu = p.get("mu", 0.05)
            sigma = p.get("sigma", 0.2)
            lam = p.get("jump_intensity", 0)
            jm = p.get("jump_mean", 0)
            js = p.get("jump_std", 0.05)
            z = self.rng.standard_normal()
            jump = 0.0
            if lam > 0:
                nj = self.rng.poisson(lam * self.dt)
                if nj > 0:
                    jump = np.sum(self.rng.normal(jm, js, nj))
            drift = (mu - 0.5 * sigma ** 2) * self.dt
            prices[t + 1] = prices[t] * np.exp(drift + sigma * np.sqrt(self.dt) * z + jump)
        return prices, regimes

    @staticmethod
    def default_regimes() -> Tuple[List[Dict], np.ndarray]:
        params = [
            {"mu": 0.10, "sigma": 0.12, "jump_intensity": 0.5, "jump_mean": 0.01, "jump_std": 0.02},  # bull
            {"mu": -0.05, "sigma": 0.25, "jump_intensity": 3.0, "jump_mean": -0.03, "jump_std": 0.05},  # bear
            {"mu": 0.02, "sigma": 0.15, "jump_intensity": 1.0, "jump_mean": 0.0, "jump_std": 0.03},   # sideways
            {"mu": -0.15, "sigma": 0.40, "jump_intensity": 5.0, "jump_mean": -0.05, "jump_std": 0.08},  # crisis
        ]
        P = np.array([
            [0.95, 0.03, 0.015, 0.005],
            [0.05, 0.90, 0.03, 0.02],
            [0.04, 0.04, 0.90, 0.02],
            [0.10, 0.10, 0.10, 0.70],
        ])
        return params, P


# ---------------------------------------------------------------------------
# 5. Order Book Simulation
# ---------------------------------------------------------------------------

@dataclass
class OrderBookLevel:
    price: float
    size: float


class OrderBookSimulator:
    """Simulated limit order book with Poisson arrivals."""

    def __init__(self, mid_price: float = 100.0, tick_size: float = 0.01,
                 n_levels: int = 10, arrival_rate: float = 100.0,
                 cancel_rate: float = 50.0,
                 rng: Optional[np.random.Generator] = None):
        self.mid = mid_price
        self.tick = tick_size
        self.n_levels = n_levels
        self.arrival_rate = arrival_rate
        self.cancel_rate = cancel_rate
        self.rng = rng or np.random.default_rng(42)
        self.bids: List[OrderBookLevel] = []
        self.asks: List[OrderBookLevel] = []
        self._init_book()

    def _init_book(self) -> None:
        self.bids = []
        self.asks = []
        for i in range(self.n_levels):
            bp = self.mid - (i + 1) * self.tick
            ap = self.mid + (i + 1) * self.tick
            bs = self.rng.exponential(1000)
            as_ = self.rng.exponential(1000)
            self.bids.append(OrderBookLevel(bp, bs))
            self.asks.append(OrderBookLevel(ap, as_))

    def step(self, dt: float = 0.001) -> Dict[str, Any]:
        """Advance order book by dt seconds."""
        events = []
        # Arrivals
        n_arrivals = self.rng.poisson(self.arrival_rate * dt)
        for _ in range(n_arrivals):
            side = self.rng.choice(["bid", "ask"])
            level = self.rng.integers(0, self.n_levels)
            size = self.rng.exponential(500)
            if side == "bid" and level < len(self.bids):
                self.bids[level] = OrderBookLevel(self.bids[level].price,
                                                   self.bids[level].size + size)
            elif side == "ask" and level < len(self.asks):
                self.asks[level] = OrderBookLevel(self.asks[level].price,
                                                   self.asks[level].size + size)
            events.append({"type": "arrival", "side": side, "level": level, "size": size})
        # Cancellations
        n_cancels = self.rng.poisson(self.cancel_rate * dt)
        for _ in range(n_cancels):
            side = self.rng.choice(["bid", "ask"])
            level = self.rng.integers(0, self.n_levels)
            if side == "bid" and level < len(self.bids):
                cancel_size = min(self.rng.exponential(300), self.bids[level].size)
                self.bids[level] = OrderBookLevel(self.bids[level].price,
                                                   max(self.bids[level].size - cancel_size, 0))
            elif side == "ask" and level < len(self.asks):
                cancel_size = min(self.rng.exponential(300), self.asks[level].size)
                self.asks[level] = OrderBookLevel(self.asks[level].price,
                                                   max(self.asks[level].size - cancel_size, 0))
        return {
            "best_bid": self.bids[0].price if self.bids else 0,
            "best_ask": self.asks[0].price if self.asks else 0,
            "spread": (self.asks[0].price - self.bids[0].price) if self.bids and self.asks else 0,
            "bid_depth": sum(b.size for b in self.bids),
            "ask_depth": sum(a.size for a in self.asks),
            "n_events": len(events),
        }

    def simulate_session(self, duration: float = 3600.0,
                          dt: float = 0.1) -> List[Dict[str, Any]]:
        n_steps = int(duration / dt)
        snapshots = []
        for _ in range(n_steps):
            snap = self.step(dt)
            snapshots.append(snap)
        return snapshots


# ---------------------------------------------------------------------------
# 6. Market Impact Model
# ---------------------------------------------------------------------------

class MarketImpact:
    """Permanent and temporary market impact models."""

    def __init__(self, sigma: float = 0.02, adv: float = 1e7,
                 permanent_coeff: float = 0.1, temporary_coeff: float = 0.05):
        self.sigma = sigma
        self.adv = adv
        self.perm = permanent_coeff
        self.temp = temporary_coeff

    def sqrt_model(self, trade_value: float) -> Dict[str, float]:
        participation = abs(trade_value) / (self.adv + 1e-10)
        permanent = self.perm * self.sigma * np.sign(trade_value) * np.sqrt(participation)
        temporary = self.temp * self.sigma * np.sqrt(participation)
        return {
            "permanent_impact": float(permanent),
            "temporary_impact": float(temporary),
            "total_cost": float(abs(trade_value) * (abs(permanent) + temporary)),
            "participation_rate": float(participation),
        }

    def almgren_chriss(self, trade_value: float, T: float = 1.0,
                       risk_aversion: float = 1e-6) -> Dict[str, Any]:
        """Optimal execution trajectory via Almgren-Chriss."""
        n_steps = max(int(T * 252), 1)
        X = abs(trade_value)
        dt_exec = T / n_steps
        # Optimal TWAP-like with urgency
        kappa = np.sqrt(risk_aversion * self.sigma ** 2 / (self.temp + 1e-10))
        trajectory = np.zeros(n_steps + 1)
        trajectory[0] = X
        for i in range(1, n_steps + 1):
            t_remain = (n_steps - i) * dt_exec
            trajectory[i] = X * np.sinh(kappa * t_remain) / (np.sinh(kappa * T) + 1e-30)
        trades = -np.diff(trajectory)
        costs = self.temp * self.sigma * np.sqrt(np.abs(trades) / (self.adv * dt_exec + 1e-10))
        return {
            "trajectory": trajectory,
            "trades": trades,
            "per_step_cost": costs,
            "total_cost": float(np.sum(costs * np.abs(trades))),
        }


# ---------------------------------------------------------------------------
# 7. Informed vs Noise Trader Flow
# ---------------------------------------------------------------------------

class InformedNoiseTraderModel:
    """Model informed and noise trader order flow."""

    def __init__(self, pct_informed: float = 0.2, sigma_noise: float = 0.02,
                 signal_strength: float = 0.05,
                 rng: Optional[np.random.Generator] = None):
        self.pct_informed = pct_informed
        self.sigma_noise = sigma_noise
        self.signal_strength = signal_strength
        self.rng = rng or np.random.default_rng(42)

    def simulate_flow(self, true_value: np.ndarray, n_trades: int = 1000) -> Dict[str, np.ndarray]:
        """
        true_value: (T,) true value path.
        Returns order flow decomposition.
        """
        T = len(true_value)
        n_per_period = n_trades // T
        order_flow = np.zeros(T)
        informed_flow = np.zeros(T)
        noise_flow = np.zeros(T)
        for t in range(T):
            for _ in range(n_per_period):
                is_informed = self.rng.random() < self.pct_informed
                if is_informed:
                    direction = np.sign(true_value[t] - true_value[max(0, t - 1)])
                    if direction == 0:
                        direction = self.rng.choice([-1, 1])
                    size = abs(self.rng.normal(self.signal_strength, self.signal_strength * 0.3))
                    informed_flow[t] += direction * size
                else:
                    size = self.rng.normal(0, self.sigma_noise)
                    noise_flow[t] += size
            order_flow[t] = informed_flow[t] + noise_flow[t]
        return {
            "total_flow": order_flow,
            "informed_flow": informed_flow,
            "noise_flow": noise_flow,
            "signal_to_noise": np.std(informed_flow) / (np.std(noise_flow) + 1e-10),
        }


# ---------------------------------------------------------------------------
# 8. Flash Crash Simulation
# ---------------------------------------------------------------------------

class FlashCrashSimulator:
    """Simulate flash crash: sudden liquidity withdrawal."""

    def __init__(self, normal_vol: float = 0.15, crash_vol: float = 0.80,
                 recovery_speed: float = 0.3, crash_depth: float = -0.10,
                 rng: Optional[np.random.Generator] = None):
        self.normal_vol = normal_vol
        self.crash_vol = crash_vol
        self.recovery_speed = recovery_speed
        self.crash_depth = crash_depth
        self.rng = rng or np.random.default_rng(42)

    def simulate(self, S0: float, n_steps: int, crash_start: int,
                 crash_duration: int = 10, recovery_duration: int = 50) -> Dict[str, np.ndarray]:
        dt = 1 / 252
        prices = np.zeros(n_steps + 1)
        volumes = np.zeros(n_steps + 1)
        spreads = np.zeros(n_steps + 1)
        prices[0] = S0
        volumes[0] = 1.0
        spreads[0] = 0.01
        crash_end = crash_start + crash_duration
        recovery_end = crash_end + recovery_duration
        for t in range(n_steps):
            if crash_start <= t < crash_end:
                # Crash phase
                progress = (t - crash_start) / crash_duration
                vol = self.crash_vol
                drift = self.crash_depth / crash_duration
                volumes[t + 1] = 0.1 + 3.0 * progress
                spreads[t + 1] = 0.01 * (1 + 20 * progress)
            elif crash_end <= t < recovery_end:
                # Recovery phase
                progress = (t - crash_end) / recovery_duration
                vol = self.crash_vol * (1 - progress) + self.normal_vol * progress
                drift = -self.crash_depth * self.recovery_speed / recovery_duration
                volumes[t + 1] = 3.0 * (1 - progress) + 1.0 * progress
                spreads[t + 1] = 0.01 * (1 + 20 * (1 - progress))
            else:
                vol = self.normal_vol
                drift = 0.05 * dt
                volumes[t + 1] = 1.0 + self.rng.standard_normal() * 0.2
                spreads[t + 1] = 0.01
            z = self.rng.standard_normal()
            prices[t + 1] = prices[t] * np.exp(drift * dt + vol * np.sqrt(dt) * z)
        return {
            "prices": prices,
            "volumes": np.maximum(volumes, 0.01),
            "spreads": np.maximum(spreads, 0.001),
            "crash_window": (crash_start, crash_end),
            "recovery_window": (crash_end, recovery_end),
        }


# ---------------------------------------------------------------------------
# 9. Earnings Announcement Model
# ---------------------------------------------------------------------------

class EarningsAnnouncementModel:
    """Simulate gap + vol spike around earnings."""

    def __init__(self, gap_mean: float = 0.0, gap_std: float = 0.05,
                 vol_spike_mult: float = 3.0, vol_decay: float = 0.85,
                 rng: Optional[np.random.Generator] = None):
        self.gap_mean = gap_mean
        self.gap_std = gap_std
        self.vol_spike = vol_spike_mult
        self.vol_decay = vol_decay
        self.rng = rng or np.random.default_rng(42)

    def simulate(self, S0: float, n_days: int, announcement_days: List[int],
                 base_vol: float = 0.2) -> Dict[str, np.ndarray]:
        dt = 1 / 252
        prices = np.zeros(n_days + 1)
        realized_vol = np.zeros(n_days + 1)
        prices[0] = S0
        current_vol_mult = 1.0
        for t in range(n_days):
            if t in announcement_days:
                gap = self.rng.normal(self.gap_mean, self.gap_std)
                prices[t] *= np.exp(gap)
                current_vol_mult = self.vol_spike
            else:
                current_vol_mult = max(1.0, current_vol_mult * self.vol_decay)
            vol = base_vol * current_vol_mult
            realized_vol[t + 1] = vol
            z = self.rng.standard_normal()
            prices[t + 1] = prices[t] * np.exp(-0.5 * vol ** 2 * dt + vol * np.sqrt(dt) * z)
        return {
            "prices": prices,
            "realized_vol": realized_vol,
            "announcement_days": announcement_days,
        }


# ---------------------------------------------------------------------------
# 10. Intraday Patterns
# ---------------------------------------------------------------------------

class IntradayPatternSimulator:
    """U-shaped volume, spread widening at open/close."""

    def __init__(self, n_minutes: int = 390, base_vol: float = 0.15,
                 rng: Optional[np.random.Generator] = None):
        self.n_minutes = n_minutes  # 6.5 hours
        self.base_vol = base_vol
        self.rng = rng or np.random.default_rng(42)

    def u_shaped_volume(self) -> np.ndarray:
        """Generate U-shaped intraday volume profile."""
        t = np.linspace(0, 1, self.n_minutes)
        u_shape = 3.0 * (t - 0.5) ** 2 + 0.5
        # Add lunch dip
        lunch = 1.0 - 0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
        volume = u_shape * lunch
        volume /= volume.mean()
        return volume

    def simulate_intraday(self, S0: float = 100.0) -> Dict[str, np.ndarray]:
        volume_profile = self.u_shaped_volume()
        dt = 1.0 / (self.n_minutes * 252)
        prices = np.zeros(self.n_minutes + 1)
        spreads = np.zeros(self.n_minutes + 1)
        volumes = np.zeros(self.n_minutes + 1)
        prices[0] = S0
        t_frac = np.linspace(0, 1, self.n_minutes)
        for i in range(self.n_minutes):
            vol_adj = self.base_vol * np.sqrt(volume_profile[i])
            z = self.rng.standard_normal()
            prices[i + 1] = prices[i] * np.exp(-0.5 * vol_adj ** 2 * dt + vol_adj * np.sqrt(dt) * z)
            # Spread: wider at open and close
            spread_mult = 2.0 * abs(t_frac[i] - 0.5) + 0.5
            if t_frac[i] < 0.05 or t_frac[i] > 0.95:
                spread_mult *= 2.0
            spreads[i + 1] = 0.01 * spread_mult
            volumes[i + 1] = volume_profile[i] * (1 + 0.2 * self.rng.standard_normal())
        return {
            "prices": prices,
            "spreads": np.maximum(spreads, 0.001),
            "volumes": np.maximum(volumes, 0.01),
            "volume_profile": volume_profile,
        }


# ---------------------------------------------------------------------------
# 11. Correlation Regime Shifts
# ---------------------------------------------------------------------------

class CorrelationRegimeSimulator:
    """Simulate gradual and sudden correlation regime shifts."""

    def __init__(self, n_assets: int, rng: Optional[np.random.Generator] = None):
        self.n_assets = n_assets
        self.rng = rng or np.random.default_rng(42)

    def _make_corr(self, base_corr: float) -> np.ndarray:
        C = np.full((self.n_assets, self.n_assets), base_corr)
        np.fill_diagonal(C, 1.0)
        return C

    def gradual_shift(self, n_steps: int, corr_start: float = 0.3,
                       corr_end: float = 0.8, sigmas: Optional[np.ndarray] = None,
                       S0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Linearly shift correlations over time."""
        if sigmas is None:
            sigmas = np.full(self.n_assets, 0.2)
        if S0 is None:
            S0 = np.full(self.n_assets, 100.0)
        prices = np.zeros((n_steps + 1, self.n_assets))
        corr_path = np.zeros(n_steps + 1)
        prices[0] = S0
        dt = 1 / 252
        for t in range(n_steps):
            frac = t / max(n_steps - 1, 1)
            c = corr_start + (corr_end - corr_start) * frac
            corr_path[t + 1] = c
            C = self._make_corr(c)
            L = np.linalg.cholesky(C + np.eye(self.n_assets) * 1e-8)
            z = L @ self.rng.standard_normal(self.n_assets)
            ret = -0.5 * sigmas ** 2 * dt + sigmas * np.sqrt(dt) * z
            prices[t + 1] = prices[t] * np.exp(ret)
        return prices, corr_path

    def sudden_shift(self, n_steps: int, shift_time: int,
                      corr_before: float = 0.3, corr_after: float = 0.8,
                      sigmas: Optional[np.ndarray] = None,
                      S0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if sigmas is None:
            sigmas = np.full(self.n_assets, 0.2)
        if S0 is None:
            S0 = np.full(self.n_assets, 100.0)
        prices = np.zeros((n_steps + 1, self.n_assets))
        corr_path = np.zeros(n_steps + 1)
        prices[0] = S0
        dt = 1 / 252
        for t in range(n_steps):
            c = corr_before if t < shift_time else corr_after
            corr_path[t + 1] = c
            C = self._make_corr(c)
            L = np.linalg.cholesky(C + np.eye(self.n_assets) * 1e-8)
            z = L @ self.rng.standard_normal(self.n_assets)
            ret = -0.5 * sigmas ** 2 * dt + sigmas * np.sqrt(dt) * z
            prices[t + 1] = prices[t] * np.exp(ret)
        return prices, corr_path


# ---------------------------------------------------------------------------
# 12. Central Bank Intervention
# ---------------------------------------------------------------------------

class CentralBankIntervention:
    """Rate change shock model with market response."""

    def __init__(self, base_rate: float = 0.05, rate_vol: float = 0.0025,
                 rng: Optional[np.random.Generator] = None):
        self.base_rate = base_rate
        self.rate_vol = rate_vol
        self.rng = rng or np.random.default_rng(42)

    def simulate_rate_path(self, n_steps: int,
                            meeting_days: List[int],
                            rate_changes: Optional[List[float]] = None) -> np.ndarray:
        rates = np.zeros(n_steps + 1)
        rates[0] = self.base_rate
        if rate_changes is None:
            rate_changes = [self.rng.choice([-0.0025, 0, 0.0025], p=[0.2, 0.6, 0.2])
                            for _ in meeting_days]
        change_map = dict(zip(meeting_days, rate_changes))
        for t in range(n_steps):
            rates[t + 1] = rates[t]
            if t in change_map:
                rates[t + 1] += change_map[t]
            rates[t + 1] += self.rng.normal(0, self.rate_vol * 0.01)
        return rates

    def rate_shock_impact(self, S0: float, rate_change: float,
                           duration_sensitivity: float = -5.0,
                           equity_sensitivity: float = -10.0,
                           n_days_after: int = 30) -> Dict[str, np.ndarray]:
        """Simulate price path after a rate shock."""
        dt = 1 / 252
        prices = np.zeros(n_days_after + 1)
        prices[0] = S0
        initial_shock = equity_sensitivity * rate_change
        prices[0] *= np.exp(initial_shock)
        # Gradual adjustment with mean reversion
        vol = 0.20
        for t in range(n_days_after):
            drift = 0.5 * initial_shock * np.exp(-0.1 * t)  # gradual recovery
            z = self.rng.standard_normal()
            prices[t + 1] = prices[t] * np.exp(drift * dt + vol * np.sqrt(dt) * z)
        return {
            "prices": prices,
            "initial_shock_pct": float(initial_shock * 100),
            "rate_change": rate_change,
        }


# ---------------------------------------------------------------------------
# Full Market Simulation Pipeline
# ---------------------------------------------------------------------------

def full_market_simulation(n_assets: int = 5, n_days: int = 504,
                            include_jumps: bool = True,
                            include_regimes: bool = True,
                            include_flash_crash: bool = False,
                            seed: int = 42) -> Dict[str, Any]:
    """Run comprehensive market simulation."""
    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {}

    # Base GBM
    mus = rng.uniform(0.02, 0.12, n_assets)
    sigmas = rng.uniform(0.10, 0.35, n_assets)
    corr = np.eye(n_assets) * 0.5 + np.ones((n_assets, n_assets)) * 0.5
    np.fill_diagonal(corr, 1.0)
    gbm = CorrelatedGBM(n_assets, mus, sigmas, corr, rng=rng)
    S0 = rng.uniform(50, 200, n_assets)
    prices = gbm.simulate(S0, n_days)
    results["gbm_prices"] = prices

    # Heston overlay for first asset
    heston = HestonModel(rng=rng)
    h_prices, h_vols = heston.simulate(S0[0], n_days)
    results["heston_prices"] = h_prices
    results["heston_vols"] = h_vols

    # Jump diffusion
    if include_jumps:
        jd = MertonJumpDiffusion(rng=rng)
        jd_prices, jd_jumps = jd.simulate(S0[0], n_days)
        results["jump_prices"] = jd_prices
        results["jump_indicators"] = jd_jumps

    # Regime switching
    if include_regimes:
        params, P = RegimeSwitchingSimulator.default_regimes()
        rs = RegimeSwitchingSimulator(params, P, rng=rng)
        rs_prices, rs_regimes = rs.simulate(S0[0], n_days)
        results["regime_prices"] = rs_prices
        results["regimes"] = rs_regimes

    # Flash crash
    if include_flash_crash:
        fc = FlashCrashSimulator(rng=rng)
        fc_result = fc.simulate(S0[0], n_days, crash_start=n_days // 2)
        results["flash_crash"] = fc_result

    # Market impact
    mi = MarketImpact()
    results["impact_1M"] = mi.sqrt_model(1e6)
    results["impact_10M"] = mi.sqrt_model(1e7)

    # Intraday
    intra = IntradayPatternSimulator(rng=rng)
    results["intraday"] = intra.simulate_intraday(S0[0])

    return results
