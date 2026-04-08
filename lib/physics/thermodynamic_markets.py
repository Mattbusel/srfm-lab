"""
Market thermodynamics: price dynamics as a thermodynamic system.

Maps financial quantities to thermodynamic analogs:
  - Price → position
  - Returns → velocity
  - Volatility → temperature
  - Volume → energy/entropy
  - Market cap → internal energy
  - Liquidity → thermal conductivity
  - Momentum → heat flow
  - Mean reversion → thermal equilibration

Implements:
  - Market temperature (local volatility as thermodynamic temperature)
  - Entropy production rate
  - Equipartition theorem analog (vol allocation across assets)
  - Carnot efficiency bound (maximum alpha extractable)
  - Phase transitions (vol regime changes as phase transitions)
  - Maxwell-Boltzmann velocity distribution (return distribution)
  - Free energy (expected PnL minus entropy cost)
  - Thermal noise floor (minimum detectable signal)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Market temperature ────────────────────────────────────────────────────────

def market_temperature(
    returns: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Rolling market temperature T = (1/2) * sigma^2.
    Analog: kinetic energy per degree of freedom.
    """
    n = len(returns)
    T = np.full(n, np.nan)
    for i in range(window, n):
        sigma2 = np.var(returns[i - window: i])
        T[i] = 0.5 * sigma2
    return T


def temperature_gradient(T: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """dT/dt — rate of temperature change (vol acceleration)."""
    return np.gradient(T, dt)


# ── Maxwell-Boltzmann velocity distribution ──────────────────────────────────

def maxwell_boltzmann_pdf(v: np.ndarray, T: float, m: float = 1.0) -> np.ndarray:
    """
    Maxwell-Boltzmann speed distribution (3D).
    PDF(v) = sqrt(2/pi) * (m/kT)^{3/2} * v^2 * exp(-m*v^2/(2kT))
    Here T = market temperature (vol^2/2), m = 1.
    """
    k_T = T  # kT = T for k=1
    if k_T <= 0:
        return np.zeros_like(v)
    coeff = math.sqrt(2 / math.pi) * (m / k_T) ** 1.5
    return coeff * v ** 2 * np.exp(-m * v ** 2 / (2 * k_T))


def return_distribution_fit(
    returns: np.ndarray,
) -> dict:
    """
    Fit returns to Maxwell-Boltzmann analog and measure deviation.
    Deviation from MB → non-equilibrium market (trending or crisis).
    """
    from scipy.stats import kstest

    T = float(np.var(returns) / 2)
    abs_returns = np.abs(returns)

    v_grid = np.linspace(0, abs_returns.max() * 1.5, 200)
    pdf_mb = maxwell_boltzmann_pdf(v_grid, T)

    # Empirical CDF vs MB CDF
    def mb_cdf(x):
        # Numerical integration of MB PDF
        from scipy.integrate import quad
        result, _ = quad(lambda v: maxwell_boltzmann_pdf(np.array([v]), T)[0], 0, x)
        return result

    # KS test: are absolute returns MB distributed?
    # We use chi-squared instead (simpler)
    hist, edges = np.histogram(abs_returns, bins=20, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    pdf_expected = maxwell_boltzmann_pdf(centers, T)

    # Chi-squared deviation
    chi2 = float(np.sum((hist - pdf_expected) ** 2 / (pdf_expected + 1e-10)))

    return {
        "temperature": T,
        "equilibrium_deviation": chi2,
        "is_equilibrium": bool(chi2 < 10.0),
        "skewness": float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-10)) ** 3)),
        "excess_kurtosis": float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-10)) ** 4) - 3),
    }


# ── Entropy production ────────────────────────────────────────────────────────

def entropy_production_rate(
    returns: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Entropy production rate: dS/dt ~ sigma^2 / T_env.
    In markets: how rapidly information entropy is being created.
    High rate → regime change / high uncertainty.
    Low rate → stable/predictable.
    """
    T = market_temperature(returns, window)
    sigma = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        sigma[i] = np.std(returns[i - window: i])

    # dS/dt ~ sigma^2 / T (irreversible entropy production)
    rate = np.where(T > 0, 2 * T / (T + 1e-10), np.nan)  # simplified
    return rate


# ── Carnot efficiency bound ───────────────────────────────────────────────────

def carnot_efficiency(T_hot: float, T_cold: float) -> float:
    """
    Maximum efficiency of a trading strategy as a heat engine.
    eta_Carnot = 1 - T_cold/T_hot

    T_hot = temperature in high-vol regime (more energy available)
    T_cold = temperature in low-vol regime (steady state)

    Interpretation: maximum alpha extractable from vol regime difference.
    """
    if T_hot <= 0 or T_cold < 0:
        return 0.0
    return max(0.0, 1 - T_cold / T_hot)


def regime_carnot_efficiency(
    returns: np.ndarray,
    vol_threshold_pct: float = 75,
) -> dict:
    """
    Estimate Carnot bound using high/low vol periods.
    """
    vols = np.array([
        returns[max(0, i - 20): i].std()
        for i in range(20, len(returns))
    ])
    threshold = np.percentile(vols, vol_threshold_pct)
    T_hot = float((vols[vols >= threshold] ** 2 / 2).mean())
    T_cold = float((vols[vols < threshold] ** 2 / 2).mean())

    eta = carnot_efficiency(T_hot, T_cold)
    return {
        "T_hot": T_hot,
        "T_cold": T_cold,
        "carnot_efficiency": eta,
        "max_annual_sharpe": float(eta * math.sqrt(252)),  # rough bound
    }


# ── Phase transitions ─────────────────────────────────────────────────────────

@dataclass
class MarketPhase:
    GAS = "gas"           # high entropy, random walks
    LIQUID = "liquid"     # moderate order, mean-reverting
    SOLID = "solid"       # low entropy, strongly trending


def detect_market_phase(
    returns: np.ndarray,
    window: int = 30,
) -> str:
    """
    Classify market thermodynamic phase from recent returns.
    """
    if len(returns) < window:
        return MarketPhase.GAS

    r = returns[-window:]
    T = float(np.var(r))  # temperature proxy

    # Autocorrelation at lag 1 (order parameter)
    if len(r) > 2:
        ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    else:
        ac = 0.0

    # Hurst (from DFA would be ideal; use simpler proxy here)
    cum_r = np.cumsum(r - r.mean())
    if cum_r.std() > 0:
        hurst_proxy = math.log(cum_r.max() - cum_r.min() + 1e-10) / math.log(len(r))
    else:
        hurst_proxy = 0.5

    if T > np.percentile(np.abs(returns), 75) ** 2 and abs(ac) < 0.1:
        return MarketPhase.GAS
    elif hurst_proxy > 0.6:
        return MarketPhase.SOLID
    else:
        return MarketPhase.LIQUID


def phase_transition_probability(
    returns: np.ndarray,
    window: int = 20,
    threshold_z: float = 2.5,
) -> np.ndarray:
    """
    Detect impending phase transitions via critical slowing down.
    Near a phase transition: autocorrelation ↑, variance ↑.
    Returns probability score [0,1] for each bar.
    """
    n = len(returns)
    probs = np.full(n, np.nan)

    for i in range(window * 2, n):
        r = returns[i - window: i]
        var = np.var(r)
        if len(r) > 2:
            ac = abs(float(np.corrcoef(r[:-1], r[1:])[0, 1]))
        else:
            ac = 0.0

        # Historical baseline
        r_hist = returns[i - 2 * window: i - window]
        var_hist = np.var(r_hist)
        if var_hist > 0:
            var_z = (var - var_hist) / (var_hist + 1e-10)
        else:
            var_z = 0.0

        # Critical slowing down: high autocorrelation + rising variance
        score = 0.5 * min(ac / 0.5, 1.0) + 0.5 * min(max(var_z, 0) / threshold_z, 1.0)
        probs[i] = float(np.clip(score, 0, 1))

    return probs


# ── Free energy and signal quality ───────────────────────────────────────────

def trading_free_energy(
    expected_return: float,
    temperature: float,
    entropy_cost: float,
) -> float:
    """
    F = E - T*S  (Helmholtz free energy analog).
    Positive F = net energy available for extraction.
    E = expected return
    T = market temperature (vol)
    S = information entropy cost of the trade

    Only take trades with F > 0.
    """
    return expected_return - temperature * entropy_cost


def signal_thermal_noise(temperature: float, bandwidth: float = 1.0) -> float:
    """
    Johnson-Nyquist thermal noise analog.
    Minimum detectable signal power P = T * bandwidth.
    Signals below this level are indistinguishable from noise.
    """
    return temperature * bandwidth


def signal_snr(
    signal_strength: float,
    temperature: float,
    bandwidth: float = 1.0,
) -> float:
    """Signal-to-noise ratio relative to thermal noise floor."""
    noise = signal_thermal_noise(temperature, bandwidth)
    return float(signal_strength / max(noise, 1e-10))
