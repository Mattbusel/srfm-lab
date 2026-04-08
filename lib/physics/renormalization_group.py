"""
Renormalization Group (RG) theory applied to financial markets.

Implements:
  - RG flow equations for market parameters
  - Fixed points and universality classes
  - Relevant/irrelevant/marginal operators
  - RG-based regime classification
  - Coarse-graining transformations (block spin analog)
  - Beta function for market coupling constants
  - Critical exponents estimation
  - Log-periodic oscillations (LPPL) — Johansen-Ledoit-Sornette
  - Renormalization of noise (signal extraction across scales)
  - Flow diagram construction
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


# ── Coarse-Graining (Block Transformation) ────────────────────────────────────

def block_average(series: np.ndarray, block_size: int) -> np.ndarray:
    """
    Block averaging: coarse-grain by averaging blocks of `block_size` observations.
    Analog of block spin transformation in RG.
    """
    n = len(series) // block_size
    return np.array([series[i * block_size: (i + 1) * block_size].mean() for i in range(n)])


def rg_flow_moments(
    returns: np.ndarray,
    scales: list[int],
) -> dict:
    """
    Compute moments at different coarse-graining scales.
    Tracks how distribution properties (mean, variance, kurtosis) flow under RG.

    Fixed point: moments stop changing → scale-invariant distribution.
    """
    flow = []
    for s in scales:
        coarse = block_average(returns, s)
        if len(coarse) < 4:
            continue
        # Normalize by sqrt(s) (CLT expectation)
        normalized = coarse / math.sqrt(s) if s > 1 else coarse
        mu = float(normalized.mean())
        var = float(normalized.var())
        skew = float(((normalized - mu)**3).mean() / (normalized.std()**3 + 1e-10))
        kurt = float(((normalized - mu)**4).mean() / (normalized.var()**2 + 1e-10))

        flow.append({
            "scale": s,
            "mean": mu,
            "variance": var,
            "skewness": skew,
            "excess_kurtosis": kurt - 3,
            "n_obs": len(coarse),
        })

    # Identify fixed point: variance stabilizes near 1 (Gaussian fixed point)
    if len(flow) >= 2:
        var_flow = [f["variance"] for f in flow]
        variance_converging = bool(abs(var_flow[-1] - 1.0) < abs(var_flow[0] - 1.0))
        kurt_flow = [f["excess_kurtosis"] for f in flow]
        towards_gaussian = bool(abs(kurt_flow[-1]) < abs(kurt_flow[0]))
    else:
        variance_converging = False
        towards_gaussian = False

    return {
        "flow": flow,
        "fixed_point_type": "Gaussian" if towards_gaussian else "Non-Gaussian",
        "variance_converging": variance_converging,
        "towards_gaussian": towards_gaussian,
    }


# ── Beta Function ──────────────────────────────────────────────────────────────

def rg_beta_function(
    coupling: np.ndarray,   # coupling constant at different scales
    log_scales: np.ndarray, # log of the coarse-graining scale
) -> np.ndarray:
    """
    Beta function: β(g) = dg/d(log μ)
    Positive β: coupling grows at large scales (IR relevant)
    Negative β: coupling shrinks (IR irrelevant, UV relevant)
    Zero: fixed point
    """
    if len(coupling) < 2:
        return np.zeros(len(coupling))
    return np.gradient(coupling, log_scales)


def classify_operator(
    coupling_flow: np.ndarray,
    scale_flow: np.ndarray,
) -> str:
    """
    Classify coupling as relevant, irrelevant, or marginal under RG flow.
    Based on whether coupling grows, shrinks, or stays constant as scale increases.
    """
    beta = rg_beta_function(coupling_flow, np.log(scale_flow + 1e-10))
    avg_beta = float(beta.mean())

    if avg_beta > 0.1:
        return "relevant"
    elif avg_beta < -0.1:
        return "irrelevant"
    else:
        return "marginal"


# ── Critical Exponents ────────────────────────────────────────────────────────

def estimate_critical_exponents(
    returns: np.ndarray,
    scales: Optional[list[int]] = None,
) -> dict:
    """
    Estimate scaling exponents from multiscale moment analysis.
    ζ(q) = q*H for monofractal, nonlinear for multifractal.

    Critical exponents:
    - η (anomalous dimension): deviation from Gaussian scaling
    - ν (correlation length): divergence near critical point
    - β (order parameter): how order parameter vanishes at transition
    """
    n = len(returns)
    if scales is None:
        scales = [2**k for k in range(1, int(math.log2(n // 4)) + 1)]

    q_values = [1, 2, 3, 4]
    structure_functions = {q: [] for q in q_values}
    scale_list = []

    for s in scales:
        coarse = block_average(np.abs(returns), s)
        if len(coarse) < 4:
            continue
        scale_list.append(s)
        for q in q_values:
            structure_functions[q].append(float(np.mean(coarse**q)))

    if len(scale_list) < 2:
        return {"eta": 0.0, "nu": 1.0, "zeta": {}}

    log_scales = np.log(scale_list)
    zeta = {}
    for q in q_values:
        sf = np.array(structure_functions[q])
        if (sf > 0).all():
            log_sf = np.log(sf)
            z, _ = np.polyfit(log_scales, log_sf, 1), None
            zeta[q] = float(z[0])
        else:
            zeta[q] = float(q) * 0.5

    # Hurst from q=2 scaling
    H_est = float(zeta.get(2, 1.0) / 2)

    # Anomalous dimension: η = 2 - zeta(2) (deviation from mean-field)
    eta = float(2 - zeta.get(2, 2.0))

    # Correlation length exponent from variance scaling
    nu = float(1.0 / max(abs(eta), 0.1))

    # Multifractality: nonlinearity of ζ(q)
    linear_zeta = {q: q * H_est for q in q_values}
    nonlinearity = float(
        sum(abs(zeta.get(q, q * H_est) - linear_zeta[q]) for q in q_values) / len(q_values)
    )

    return {
        "hurst": H_est,
        "eta": eta,
        "nu": nu,
        "zeta": zeta,
        "multifractality": nonlinearity,
        "is_multifractal": nonlinearity > 0.05,
        "universality_class": (
            "Mean field (Gaussian)" if abs(eta) < 0.1 else
            "Ising-like" if abs(eta - 0.25) < 0.15 else
            "Lévy/heavy-tail"
        ),
    }


# ── Log-Periodic Power Law (LPPL) ─────────────────────────────────────────────

@dataclass
class LPPLParams:
    A: float = 0.0     # log-price at crash
    B: float = -0.1    # power law amplitude
    C: float = 0.01    # oscillation amplitude
    tc: float = 1.0    # critical time
    m: float = 0.45    # power law exponent (0 < m < 1)
    omega: float = 10.0  # log-frequency
    phi: float = 0.0   # phase


def lppl_formula(t: np.ndarray, p: LPPLParams) -> np.ndarray:
    """LPPL: log(P) = A + B*(tc-t)^m + C*(tc-t)^m * cos(omega*log(tc-t) + phi)"""
    dt = np.maximum(p.tc - t, 1e-8)
    return p.A + p.B * dt**p.m + p.C * dt**p.m * np.cos(p.omega * np.log(dt) + p.phi)


def fit_lppl(
    log_prices: np.ndarray,
    t: np.ndarray,
    n_restarts: int = 20,
) -> tuple[LPPLParams, float]:
    """
    Fit LPPL model to log-prices (Johansen-Ledoit-Sornette bubble model).
    Returns (params, residual_rmse).
    """
    from scipy.optimize import minimize

    def objective(x):
        p = LPPLParams(A=x[0], B=x[1], C=x[2], tc=x[3], m=x[4], omega=x[5], phi=x[6])
        if p.tc <= t[-1] or p.m <= 0 or p.m >= 1 or p.omega < 1:
            return 1e10
        pred = lppl_formula(t, p)
        return float(np.mean((log_prices - pred)**2))

    rng = np.random.default_rng(42)
    best_rmse = np.inf
    best_params = LPPLParams()

    t_range = t[-1] - t[0]
    price_range = log_prices[-1] - log_prices[0]

    for _ in range(n_restarts):
        x0 = [
            log_prices[-1] + rng.uniform(0, 0.2) * abs(price_range),
            rng.uniform(-0.5, 0.0),
            rng.uniform(-0.05, 0.05),
            t[-1] + rng.uniform(0.01, 0.5) * t_range,
            rng.uniform(0.1, 0.9),
            rng.uniform(5, 15),
            rng.uniform(0, 2 * math.pi),
        ]
        try:
            res = minimize(objective, x0, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-6})
            if res.fun < best_rmse:
                best_rmse = res.fun
                p = LPPLParams(*res.x)
                if p.tc > t[-1] and 0 < p.m < 1:
                    best_params = p
        except Exception:
            pass

    return best_params, float(math.sqrt(best_rmse))


def lppl_crash_probability(
    params: LPPLParams,
    current_t: float,
    hazard_scale: float = 1.0,
) -> dict:
    """
    Estimate crash probability from LPPL fit.
    Higher oscillation amplitude C and approaching tc → higher probability.
    """
    time_to_critical = max(params.tc - current_t, 1e-6)
    dt_norm = time_to_critical / max(params.tc, 1e-6)

    # Crash indicator: 1/time_to_critical weighted by amplitude
    amplitude_signal = abs(params.C) / (abs(params.B) + 1e-10)
    time_pressure = 1.0 / max(time_to_critical, 0.01)
    raw_prob = float(math.tanh(amplitude_signal * time_pressure * hazard_scale))

    return {
        "crash_probability": raw_prob,
        "time_to_critical": float(time_to_critical),
        "amplitude_ratio": float(amplitude_signal),
        "high_crash_risk": bool(raw_prob > 0.6 and time_to_critical < 0.1),
        "lppl_quality": bool(abs(params.C) < abs(params.B) and 0.3 < params.m < 0.8),
    }


# ── RG Fixed Point Classifier ──────────────────────────────────────────────────

def find_rg_fixed_points(
    returns: np.ndarray,
    scales: list[int],
) -> dict:
    """
    Identify fixed points in the RG flow of return distributions.
    Fixed points correspond to scale-invariant market regimes.
    """
    flow = rg_flow_moments(returns, scales)["flow"]
    if len(flow) < 3:
        return {"fixed_points": [], "basin_of_attraction": "unknown"}

    # Track convergence of key quantities
    kurtoses = [f["excess_kurtosis"] for f in flow]
    variances = [f["variance"] for f in flow]

    fixed_points = []

    # Gaussian fixed point: kurtosis → 0, variance → 1
    kurt_converging = bool(abs(kurtoses[-1]) < abs(kurtoses[0]) * 0.5)
    var_stable = bool(abs(variances[-1] - 1.0) < 0.3)

    if kurt_converging:
        fixed_points.append({
            "type": "Gaussian",
            "stability": "stable",
            "kurtosis_at_fp": 0.0,
            "physical_interpretation": "CLT fixed point — diffusive regime",
        })

    # Non-Gaussian stable fixed point: kurtosis doesn't decay
    if not kurt_converging and abs(kurtoses[-1]) > 1.0:
        alpha_stable = 2.0 / (1 + abs(kurtoses[-1]) / 6)  # rough estimate
        fixed_points.append({
            "type": f"Lévy-Stable (alpha≈{alpha_stable:.2f})",
            "stability": "stable",
            "excess_kurtosis_at_fp": float(kurtoses[-1]),
            "physical_interpretation": "Fat-tail fixed point — heavy-tail regime",
        })

    basin = "Gaussian" if kurt_converging else "Lévy-Stable"

    return {
        "fixed_points": fixed_points,
        "basin_of_attraction": basin,
        "kurtosis_flow": kurtoses,
        "variance_flow": variances,
        "rg_trajectory": flow,
    }


# ── Renormalization of Noise ───────────────────────────────────────────────────

def multiscale_snr(
    signal: np.ndarray,
    noise_estimate: Optional[np.ndarray] = None,
    scales: Optional[list[int]] = None,
) -> dict:
    """
    Signal-to-noise ratio across multiple scales.
    Under RG, noise renormalizes differently than signal.
    Pure noise: SNR constant across scales.
    Signal present: SNR improves at larger scales (noise averages away).
    """
    if scales is None:
        scales = [1, 2, 5, 10, 20, 50]

    snr_by_scale = {}
    for s in scales:
        coarse = block_average(signal, s)
        if len(coarse) < 2:
            continue
        if noise_estimate is not None:
            noise_coarse = block_average(noise_estimate, s)
            snr = float(coarse.var() / (noise_coarse.var() + 1e-10))
        else:
            # Estimate noise from high-frequency residuals
            snr = float(coarse.var() * s / (signal.var() + 1e-10))
        snr_by_scale[s] = snr

    if len(snr_by_scale) < 2:
        return {"snr_by_scale": snr_by_scale, "signal_present": False}

    snr_vals = np.array(list(snr_by_scale.values()))
    improving = bool(snr_vals[-1] > snr_vals[0] * 1.2)

    return {
        "snr_by_scale": snr_by_scale,
        "signal_present": improving,
        "snr_improvement_ratio": float(snr_vals[-1] / (snr_vals[0] + 1e-10)),
        "optimal_scale": list(snr_by_scale.keys())[int(np.argmax(snr_vals))],
    }
