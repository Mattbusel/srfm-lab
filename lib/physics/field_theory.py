"""
Quantum field theory analogs for market dynamics.

Maps markets to QFT concepts:
  - Price field phi(x,t) as a scalar field
  - Returns as field fluctuations (virtual particles)
  - Volatility clustering → field self-interactions (phi^4 theory)
  - Correlations → propagator / Green's function
  - Market impact → interaction vertex
  - Renormalization group → scale-dependent effective parameters
  - Spontaneous symmetry breaking → trend onset (phi → <phi> ≠ 0)

Implements:
  - Propagator estimation from return autocorrelations
  - Effective mass parameter (inverse correlation length)
  - Running coupling constant (scale-dependent vol)
  - Spontaneous symmetry breaking detection
  - Ward-Takahashi identity analog (price conservation)
  - Feynman path integral Monte Carlo
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Field propagator ──────────────────────────────────────────────────────────

def field_propagator(
    returns: np.ndarray,
    max_lag: int = 20,
) -> np.ndarray:
    """
    Market propagator G(t-t') = <phi(t)phi(t')> (two-point function).
    Analog: returns autocorrelation function.
    Decays as G ~ exp(-m*tau) where m = effective mass.
    """
    n = len(returns)
    r_centered = returns - returns.mean()
    prop = np.correlate(r_centered, r_centered, mode="full")
    prop = prop[n - 1: n + max_lag] / prop[n - 1]  # normalize G(0)=1
    return prop


def effective_mass(propagator: np.ndarray) -> float:
    """
    Extract effective mass from propagator decay: G(tau) ~ exp(-m*tau).
    m → 0: massless field (random walk, persistent correlations)
    m large: heavy field (fast mean-reversion, short correlation length)
    """
    prop = propagator[propagator > 0]
    if len(prop) < 3:
        return 1.0
    lags = np.arange(len(prop))
    # Fit ln(G) ~ -m*tau
    log_prop = np.log(prop + 1e-12)
    slope, _ = np.polyfit(lags, log_prop, 1)
    return float(max(-slope, 0.0))


def correlation_length(m: float) -> float:
    """xi = 1/m (correlation length in time units)."""
    return 1.0 / max(m, 1e-10)


# ── Running coupling constant ─────────────────────────────────────────────────

def running_coupling(
    returns: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """
    Renormalization group: how volatility (coupling constant) changes with scale.
    lambda(mu) = sigma^2(mu) where mu = 1/window.

    Returns: volatility at each scale.
    """
    couplings = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        s = int(scale)
        if s < 2 or s > len(returns):
            continue
        # Scale-aggregated return volatility
        n_complete = len(returns) // s
        if n_complete == 0:
            continue
        agg = returns[:n_complete * s].reshape(n_complete, s).sum(axis=1)
        couplings[i] = float(agg.std())

    return couplings


def beta_function(
    couplings: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """
    Beta function: beta(lambda) = d(lambda)/d(log mu).
    beta > 0: coupling grows at smaller scales (UV unsafe)
    beta < 0: coupling shrinks (asymptotically free)
    beta = 0: fixed point (scale-invariant / fractal market)
    """
    log_scales = np.log(scales + 1e-10)
    return np.gradient(couplings, log_scales)


# ── Spontaneous symmetry breaking ─────────────────────────────────────────────

@dataclass
class SymmetryBreakingDetector:
    """
    Detect spontaneous symmetry breaking (SSB): the market develops
    a nonzero 'vacuum expectation value' <phi> ≠ 0.
    In finance: drift ≠ 0, trend onset.

    Landau-Ginzburg potential: V(phi) = -mu^2/2 * phi^2 + lambda/4 * phi^4
    For mu^2 > 0: double-well potential → SSB → trend
    For mu^2 < 0: single-well → symmetric (no trend)
    """
    window: int = 60
    order_parameter_ema: float = 0.0
    _mu2: float = -1.0  # mass parameter squared

    def update(self, returns: np.ndarray) -> dict:
        """Compute SSB indicators from recent returns."""
        r = returns[-self.window:] if len(returns) >= self.window else returns
        phi = r.mean()                   # order parameter (drift)
        sigma2 = r.var()                  # fluctuation scale

        # Landau mass parameter: mu^2 = (sigma^2_long - sigma^2_short) / sigma^2_short
        if len(returns) >= 2 * self.window:
            r_long = returns[-2 * self.window:-self.window]
            sigma2_long = r_long.var()
            mu2 = (sigma2 - sigma2_long) / max(sigma2_long, 1e-10)
        else:
            mu2 = -1.0
        self._mu2 = float(mu2)

        # Coupling constant
        r_std = max(sigma2 ** 0.5, 1e-10)
        r_normalized = r / r_std
        lambda_coupling = float(np.mean(r_normalized ** 4) - 3)  # excess kurtosis

        # SSB condition: mu^2 > 0 (positive mass → double-well)
        ssb = bool(mu2 > 0.1)

        # Vacuum expectation value (trend direction)
        vev = float(phi / r_std) if r_std > 0 else 0.0

        return {
            "order_parameter": float(phi),
            "vev": vev,
            "mu2": float(mu2),
            "lambda": float(lambda_coupling),
            "ssb_detected": ssb,
            "trend_direction": 1 if phi > 0 else (-1 if phi < 0 else 0),
            "correlation_length": float(self.window / max(abs(mu2), 0.01)),
        }


# ── Feynman path integral Monte Carlo ────────────────────────────────────────

def path_integral_mc(
    action_params: dict,
    T: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Feynman path integral approach: integrate over all price paths
    weighted by exp(-S[path]) where S is the action functional.

    Action: S = integral [(1/2)(dphi/dt)^2 + V(phi)] dt
    V(phi) = mass^2/2 * phi^2 + lambda/4 * phi^4

    This gives the quantum amplitude for price to go from phi_0 to phi_T.
    Used for: option pricing, rare event probability estimation.
    """
    rng = rng or np.random.default_rng()
    m2 = action_params.get("mass2", 1.0)
    lam = action_params.get("lambda", 0.1)
    phi0 = action_params.get("phi0", 0.0)
    dt_step = T / n_steps

    # Sample paths (random walk in phi space)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = phi0

    Z = rng.standard_normal((n_paths, n_steps))
    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] + math.sqrt(dt_step) * Z[:, i]

    # Compute action for each path
    def action(path):
        dphi = np.diff(path) / dt_step
        kinetic = 0.5 * np.sum(dphi ** 2) * dt_step
        potential = np.sum(0.5 * m2 * path ** 2 + 0.25 * lam * path ** 4) * dt_step
        return kinetic + potential

    actions = np.array([action(paths[i]) for i in range(n_paths)])
    weights = np.exp(-actions + actions.min())  # stabilize
    weights /= weights.sum()

    # Observables
    phi_T = paths[:, -1]
    expectation_phi = float(np.sum(weights * phi_T))
    expectation_phi2 = float(np.sum(weights * phi_T ** 2))

    return {
        "E_phi_T": expectation_phi,
        "E_phi2_T": expectation_phi2,
        "std_phi_T": float(math.sqrt(max(expectation_phi2 - expectation_phi ** 2, 0))),
        "partition_function": float(np.sum(np.exp(-actions + actions.min()))),
        "dominant_action": float(actions.min()),
        "tunneling_prob": float(np.mean(phi_T * phi0 < 0)),  # sign flip
    }


# ── Ward-Takahashi identity (conservation law) ───────────────────────────────

def ward_identity_check(
    prices: np.ndarray,
    timeframe: str = "daily",
) -> dict:
    """
    Check Ward-Takahashi identity analog: price is 'conserved' on average
    in an efficient market (no systematic drift in the Martingale measure).

    WT: <T[phi(x) J_mu(y)]> = 0 (no net probability current)
    In finance: E[r | F_{t-1}] = 0 (fair game condition)

    Tests: is return autocorrelation structure consistent with conservation?
    """
    r = np.diff(np.log(prices))
    n = len(r)

    # Noether current analog: probability current J = phi * dp/dt
    J = r[:-1] * r[1:]  # product of consecutive returns

    # Ward identity: <J> = 0 for martingale
    J_mean = float(J.mean())
    J_std = float(J.std() / math.sqrt(len(J)))
    J_tstat = J_mean / max(J_std, 1e-10)

    # Conserved charge: cumulative autocorrelation
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if n > 2 else 0.0

    return {
        "J_mean": J_mean,
        "J_tstat": J_tstat,
        "conservation_violated": bool(abs(J_tstat) > 2.0),
        "autocorrelation_lag1": ac1,
        "martingale_violation": bool(abs(ac1) > 0.1),
        "exploitable": bool(abs(J_tstat) > 2.0 or abs(ac1) > 0.1),
    }
