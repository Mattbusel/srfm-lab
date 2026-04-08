"""
Holographic bound and information-theoretic market capacity.

Implements:
  - Bekenstein-Hawking entropy bound for market state
  - Holographic screen (information capacity of price manifold)
  - Bousso bound (covariant entropy bound)
  - Market information content vs holographic limit
  - Entanglement entropy of correlated asset subsets
  - Mutual information capacity of the market
  - Channel capacity (Shannon) for price signal transmission
  - Holographic noise floor estimation
  - Area law for correlation entropy
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Bekenstein Bound ──────────────────────────────────────────────────────────

def bekenstein_bound(
    energy: float,     # total market energy (proxy: total market cap $)
    radius: float,     # effective radius (proxy: number of assets or degrees of freedom)
) -> float:
    """
    Bekenstein bound: S <= 2π * k * R * E / (ℏ * c)
    In natural units (ℏ = c = k = 1): S_max = 2π * R * E

    For market analog:
    - energy ≈ total dollar volume or market cap (in natural units)
    - radius ≈ effective number of independent assets
    Returns maximum information bits the market can store.
    """
    return float(2 * math.pi * radius * energy)


def market_bekenstein_bound(
    market_cap_usd: float,
    n_assets: int,
    correlation_matrix: Optional[np.ndarray] = None,
) -> dict:
    """
    Market analog of the Bekenstein bound.
    Estimates maximum information content of market state.
    """
    # Effective radius = effective number of independent assets (rank of corr)
    if correlation_matrix is not None:
        eigvals = np.linalg.eigvalsh(correlation_matrix)
        effective_n = float(eigvals.sum()**2 / (eigvals**2).sum())  # participation ratio
    else:
        effective_n = float(n_assets)

    # Energy proxy: log(market_cap) in bits
    energy_bits = math.log2(max(market_cap_usd, 1))

    s_bound = bekenstein_bound(energy_bits, effective_n)
    actual_dof = effective_n * math.log2(max(n_assets, 2))

    return {
        "bekenstein_bound_bits": float(s_bound),
        "effective_n_assets": float(effective_n),
        "actual_dof_bits": float(actual_dof),
        "saturation_ratio": float(actual_dof / max(s_bound, 1e-10)),
        "near_holographic_limit": bool(actual_dof / max(s_bound, 1e-10) > 0.8),
    }


# ── Holographic Screen ────────────────────────────────────────────────────────

@dataclass
class HolographicScreen:
    """
    Market holographic screen: information encoded on the boundary of market state space.
    Analogizes to Susskind-'t Hooft holographic principle.
    """
    n_assets: int
    n_timepoints: int
    bit_depth: int = 64   # precision bits per price observation

    @property
    def bulk_bits(self) -> float:
        """Total information in price series (bulk volume)."""
        return float(self.n_assets * self.n_timepoints * self.bit_depth)

    @property
    def screen_area(self) -> float:
        """Holographic screen area (boundary = time × assets)."""
        return float(self.n_assets + self.n_timepoints)  # linear boundary

    @property
    def screen_capacity_bits(self) -> float:
        """Maximum bits encodable on screen (1 bit per Planck area)."""
        return float(self.screen_area * self.bit_depth)

    @property
    def compression_ratio(self) -> float:
        """Bulk info / screen capacity — >1 means info exceeds holographic limit."""
        return float(self.bulk_bits / max(self.screen_capacity_bits, 1))


def holographic_redundancy(
    returns: np.ndarray,
) -> dict:
    """
    Estimate redundant information in returns matrix.
    Holographic principle says true DoF lives on boundary (lower dimension).
    Redundancy = 1 - effective_rank / max_rank.
    """
    T, N = returns.shape
    cov = np.cov(returns.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]

    # Participation ratio (PR) = effective number of factors
    pr = float(eigvals.sum()**2 / (eigvals**2).sum())
    max_rank = min(T, N)

    # Shannon entropy of eigenvalue distribution
    normalized = eigvals / eigvals.sum()
    spectral_entropy = float(-np.sum(normalized * np.log(normalized + 1e-10)))
    max_entropy = math.log(len(eigvals))

    redundancy = 1 - pr / max_rank
    info_density = spectral_entropy / max_entropy

    return {
        "effective_rank": float(pr),
        "max_rank": float(max_rank),
        "redundancy": float(redundancy),
        "spectral_entropy_bits": float(spectral_entropy / math.log(2)),
        "info_density": float(info_density),
        "holographic_compression": float(max_rank / max(pr, 1)),
    }


# ── Entanglement Entropy ──────────────────────────────────────────────────────

def entanglement_entropy(
    returns: np.ndarray,
    subset_A: list[int],
    subset_B: Optional[list[int]] = None,
) -> float:
    """
    Entanglement entropy between two subsets of assets.
    Uses mutual information as quantum entanglement proxy.
    S_A = -Tr(ρ_A log ρ_A), approximated via covariance matrix eigenvalues.
    """
    T, N = returns.shape

    if subset_B is None:
        subset_B = [i for i in range(N) if i not in subset_A]

    if not subset_A or not subset_B:
        return 0.0

    # Full covariance and reduced covariance for subset A
    Sigma = np.cov(returns.T) + 1e-8 * np.eye(N)
    Sigma_A = Sigma[np.ix_(subset_A, subset_A)]
    Sigma_B = Sigma[np.ix_(subset_B, subset_B)]

    def von_neumann_entropy(M: np.ndarray) -> float:
        """Von Neumann entropy S = -Tr(M/Tr(M) * log(M/Tr(M)))."""
        tr = float(np.trace(M))
        if tr < 1e-10:
            return 0.0
        rho = M / tr
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-10]
        return float(-np.sum(eigvals * np.log(eigvals)))

    S_A = von_neumann_entropy(Sigma_A)
    S_B = von_neumann_entropy(Sigma_B)
    S_AB = von_neumann_entropy(Sigma)

    # Mutual information (entanglement measure) = S_A + S_B - S_AB
    mutual_info = max(S_A + S_B - S_AB, 0.0)

    return float(mutual_info)


def area_law_check(
    returns: np.ndarray,
    max_subset_size: int = 10,
) -> dict:
    """
    Check if entanglement entropy follows area law (S ∝ boundary area, not volume).
    In many physical systems and low-entanglement markets, S grows with surface area.
    """
    T, N = returns.shape
    sizes = range(1, min(max_subset_size, N // 2) + 1)
    entropies = []

    rng = np.random.default_rng(42)
    for size in sizes:
        # Sample random subset of this size
        subset = rng.choice(N, size, replace=False).tolist()
        ee = entanglement_entropy(returns, subset)
        entropies.append(ee)

    entropies = np.array(entropies)
    sizes_arr = np.array(list(sizes))

    # Fit S ~ size^alpha
    log_s = np.log(entropies + 1e-10)
    log_n = np.log(sizes_arr)

    if len(log_s) >= 2:
        coeffs = np.polyfit(log_n, log_s, 1)
        alpha = float(coeffs[0])
    else:
        alpha = 1.0

    return {
        "subset_sizes": list(sizes),
        "entropies": entropies.tolist(),
        "scaling_exponent": alpha,
        "follows_area_law": bool(alpha < 0.7),   # area law: alpha ≈ 1 in 1D, 2/3 in 2D
        "volume_law": bool(alpha > 1.3),
        "interpretation": (
            "area law (low entanglement, quasi-local correlations)"
            if alpha < 0.7 else
            "volume law (high entanglement, strongly correlated market)"
        ),
    }


# ── Channel Capacity ──────────────────────────────────────────────────────────

def shannon_channel_capacity(
    snr_db: float,
    bandwidth: float = 1.0,
) -> float:
    """
    Shannon-Hartley theorem: C = B * log2(1 + SNR)
    Applied to market: how much information can price signals carry?

    snr_db: signal-to-noise ratio in dB
    bandwidth: effective trading frequency (cycles per time unit)
    Returns capacity in bits per time unit.
    """
    snr = 10 ** (snr_db / 10)
    return float(bandwidth * math.log2(1 + snr))


def market_channel_capacity(
    returns: np.ndarray,
    signal_component: np.ndarray,
    window: int = 252,
) -> dict:
    """
    Estimate market channel capacity for transmitting price signals.
    Signal = predictable component, Noise = residual.
    """
    T = min(len(returns), len(signal_component))
    noise = returns[:T] - signal_component[:T]

    signal_power = float(signal_component[:T].var())
    noise_power = float(noise.var())
    snr = signal_power / max(noise_power, 1e-10)
    snr_db = float(10 * math.log10(snr + 1e-10))

    capacity = shannon_channel_capacity(snr_db)

    # Effective bandwidth from autocorrelation
    acf = np.correlate(returns[:T] - returns[:T].mean(), returns[:T] - returns[:T].mean(), mode="full")
    acf = acf[T - 1:] / acf[T - 1]
    bandwidth = float(1.0 / max(np.searchsorted(-np.abs(acf[1:]), -0.1), 1))

    return {
        "snr_linear": float(snr),
        "snr_db": snr_db,
        "channel_capacity_bits": float(capacity),
        "effective_bandwidth": float(bandwidth),
        "information_efficiency": float(min(capacity / (math.log2(T) + 1e-10), 1.0)),
    }


# ── Holographic Noise Floor ────────────────────────────────────────────────────

def holographic_noise_floor(
    n_assets: int,
    n_observations: int,
    bit_precision: int = 64,
) -> dict:
    """
    Minimum achievable noise floor given holographic information bound.
    Below this floor, "signals" are likely artifacts of finite information capacity.
    """
    # Total information budget
    total_bits = n_observations * math.log2(max(n_assets, 2))

    # Information per observation
    bits_per_obs = total_bits / max(n_observations, 1)

    # Minimum detectable signal amplitude (noise floor from information uncertainty)
    # Heisenberg-like: Δx * Δp >= ℏ/2 → in market: ΔR * ΔT >= 1/bits
    min_detectable_return = 1.0 / max(bits_per_obs, 1)

    # Minimum R² achievable by any model given this information
    min_r_squared = 1.0 / max(n_observations / n_assets, 1)

    return {
        "total_information_bits": float(total_bits),
        "bits_per_observation": float(bits_per_obs),
        "min_detectable_return": float(min_detectable_return),
        "min_meaningful_r_squared": float(min_r_squared),
        "holographic_resolution": float(1.0 / max(bits_per_obs, 1)),
        "overfitting_threshold_params": int(n_observations // 10),
    }
