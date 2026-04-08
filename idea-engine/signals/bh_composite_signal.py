"""
Black Hole composite signal — full BH physics-inspired trading signal.

Integrates all three BH physics layers:
  - Schwarzschild/Kerr: gravitational effects (mass, spin, ergosphere)
  - Thermodynamic: entropy, temperature, phase transitions, Carnot efficiency
  - Field theory: propagator decay, symmetry breaking, path integral

Generates a unified trading signal from all physics layers combined.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BHCompositeParams:
    # Layer weights
    gravitational_weight: float = 0.35
    thermodynamic_weight: float = 0.35
    field_theory_weight: float = 0.30

    # Gravitational params
    bh_mass_window: int = 50          # window for BH mass estimation
    kerr_spin_window: int = 20        # window for angular momentum

    # Thermodynamic params
    temp_window: int = 30
    entropy_threshold: float = 0.7    # high entropy = chaotic regime

    # Field theory params
    correlation_length_window: int = 40
    ssb_detection_window: int = 60

    # Signal params
    signal_entry_threshold: float = 0.5
    signal_exit_threshold: float = 0.2


@dataclass
class BHSignalComponents:
    gravitational: float         # -1 to +1
    thermodynamic: float         # -1 to +1
    field_theory: float          # -1 to +1
    composite: float             # -1 to +1
    regime: str
    confidence: float
    warnings: list[str] = field(default_factory=list)


def _bh_mass(returns: np.ndarray, window: int) -> float:
    """BH mass = cumulative momentum (absolute drift)."""
    n = min(len(returns), window)
    return float(abs(returns[-n:].mean()) * math.sqrt(n))


def _angular_momentum(prices: np.ndarray, window: int) -> float:
    """Kerr spin = rotational momentum of price = tendency to keep rotating."""
    n = min(len(prices), window)
    p = prices[-n:]
    returns = np.diff(np.log(p))
    # Angular momentum proxy: persistence of direction
    if len(returns) < 2:
        return 0.0
    sign_changes = np.sum(np.diff(np.sign(returns)) != 0)
    persistence = 1 - 2 * sign_changes / max(len(returns) - 1, 1)
    return float(persistence * abs(returns).mean() * math.sqrt(n))


def _ergosphere_signal(mass: float, spin: float, price: float) -> float:
    """
    Energy extraction from ergosphere (Penrose process analog).
    High spin + low mass → ergosphere signal (+1 long, -1 short).
    """
    spin_ratio = abs(spin) / max(mass + abs(spin), 1e-10)
    if spin_ratio > 0.7:
        return float(math.copysign(1.0, spin))  # extract energy in spin direction
    return 0.0


def _market_temperature(returns: np.ndarray, window: int) -> float:
    """Market temperature = realized volatility (thermal energy proxy)."""
    n = min(len(returns), window)
    return float(returns[-n:].std() * math.sqrt(252))


def _entropy_signal(returns: np.ndarray, window: int) -> float:
    """
    Information entropy of returns. High entropy = chaotic → stay out.
    Low entropy = ordered → trend or MR setup.
    """
    n = min(len(returns), window)
    r = returns[-n:]
    # Discretize
    n_bins = max(5, n // 10)
    hist, _ = np.histogram(r, bins=n_bins)
    probs = hist / (hist.sum() + 1e-10)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = math.log(n_bins)
    normalized = entropy / max(max_entropy, 1e-10)

    # Low entropy = order → signal in trend direction
    if normalized < 0.4:
        trend = float(r.mean() / (r.std() + 1e-10))
        return float(math.tanh(trend))
    elif normalized > 0.8:
        return 0.0  # too chaotic
    else:
        # Moderate: slight trend-follow
        trend = float(r.mean() / (r.std() + 1e-10))
        return float(math.tanh(trend) * (1 - normalized))


def _phase_transition_signal(returns: np.ndarray, window: int) -> float:
    """
    Phase transition detection: near critical point → reduce position.
    At critical point: diverging fluctuations.
    """
    n = min(len(returns), window)
    r = returns[-n:]
    # Susceptibility: variance of |return| (diverges at transition)
    susceptibility = float(np.abs(r).var())
    # Correlation length: how far correlations extend
    if n >= 10:
        acf = float(np.corrcoef(r[1:], r[:-1])[0, 1])
        corr_length = abs(acf) / max(1 - abs(acf), 0.01)
    else:
        corr_length = 1.0

    # Near critical: high susceptibility and correlation length
    critical_proximity = float(math.tanh(susceptibility * corr_length * 100))

    # At criticality: contrarian signal
    trend = float(r.mean() / (r.std() + 1e-10))
    return float(-math.copysign(critical_proximity * 0.5, trend))


def _field_propagator_signal(returns: np.ndarray, window: int) -> float:
    """
    Field theory propagator: how quickly correlations decay.
    Short decay = mass gap → bounded oscillations (MR signal).
    Long decay = massless = trending.
    """
    n = min(len(returns), window)
    r = returns[-n:]
    if n < 10:
        return 0.0

    # Correlation function at various lags
    lags = range(1, min(10, n // 4))
    acfs = []
    for lag in lags:
        if len(r) > lag:
            acf = float(np.corrcoef(r[lag:], r[:-lag])[0, 1])
            acfs.append((lag, acf))

    if not acfs:
        return 0.0

    # Fit exponential decay: C(t) ~ exp(-t/xi)
    lags_arr = np.array([a[0] for a in acfs])
    acfs_arr = np.abs(np.array([a[1] for a in acfs]))

    if (acfs_arr > 0).any() and acfs_arr[0] > 0.05:
        log_acf = np.log(acfs_arr + 1e-10)
        try:
            slope = float(np.polyfit(lags_arr, log_acf, 1)[0])
            decay_rate = abs(slope)
        except Exception:
            decay_rate = 1.0
    else:
        decay_rate = 1.0

    # Fast decay = MR signal, slow decay = momentum signal
    trend = float(r.mean() / (r.std() + 1e-10))
    if decay_rate > 0.5:
        # Short correlation length → mean reversion
        return float(-math.tanh(trend) * min(decay_rate, 1.0))
    else:
        # Long correlation length → trend
        return float(math.tanh(trend) * min(1.0 / max(decay_rate, 0.1) * 0.2, 1.0))


def _symmetry_breaking_signal(returns: np.ndarray, prices: np.ndarray, window: int) -> float:
    """
    Spontaneous symmetry breaking: mean of return distribution shifts.
    Detects when market transitions to directional bias.
    """
    n = min(len(returns), window)
    r = returns[-n:]
    if n < 20:
        return 0.0

    # Compare early half vs late half
    half = n // 2
    early_mean = float(r[:half].mean())
    late_mean = float(r[half:].mean())
    combined_std = float(r.std())

    # T-test for mean shift
    mean_shift = float((late_mean - early_mean) / max(combined_std * math.sqrt(2 / half), 1e-10))

    # SSB signal: bias in direction of shift
    if abs(mean_shift) > 1.5:
        return float(math.tanh(mean_shift) * 0.8)
    return 0.0


def compute_bh_composite(
    prices: np.ndarray,
    volume: Optional[np.ndarray] = None,
    params: Optional[BHCompositeParams] = None,
) -> BHSignalComponents:
    """
    Compute full Black Hole composite signal from price (and optionally volume) data.
    """
    if params is None:
        params = BHCompositeParams()

    if len(prices) < 20:
        return BHSignalComponents(
            gravitational=0.0, thermodynamic=0.0, field_theory=0.0,
            composite=0.0, regime="insufficient_data", confidence=0.0,
        )

    returns = np.diff(np.log(prices))
    warnings = []

    # ── Gravitational Layer ───────────────────────────────────────────────────
    mass = _bh_mass(returns, params.bh_mass_window)
    spin = _angular_momentum(prices, params.kerr_spin_window)
    ergo = _ergosphere_signal(mass, spin, prices[-1])

    # Gravitational lensing: price distortion near high-mass event
    recent_trend = float(returns[-min(5, len(returns)):].mean() /
                         (returns[-min(20, len(returns)):].std() + 1e-10))
    grav_signal = float(0.5 * math.tanh(recent_trend * 2) + 0.5 * ergo)

    if mass > returns[-params.bh_mass_window:].std() * 3:
        warnings.append("Very high BH mass: strong directional momentum")

    # ── Thermodynamic Layer ───────────────────────────────────────────────────
    temp = _market_temperature(returns, params.temp_window)
    entropy_sig = _entropy_signal(returns, params.temp_window)
    phase_sig = _phase_transition_signal(returns, params.temp_window)

    # Carnot efficiency proxy: how much free energy available for trading
    temp_baseline = float(returns.std() * math.sqrt(252))
    if temp_baseline > 0:
        temp_ratio = min(temp / temp_baseline, 3.0)
    else:
        temp_ratio = 1.0

    carnot_eff = max(1 - 1 / max(temp_ratio, 1.01), 0)
    thermo_signal = float(entropy_sig * (1 - carnot_eff * 0.3) + phase_sig * 0.3)
    thermo_signal = float(np.clip(thermo_signal, -1, 1))

    if temp > temp_baseline * 2.5:
        warnings.append("Very high market temperature: elevated volatility")

    # ── Field Theory Layer ────────────────────────────────────────────────────
    prop_signal = _field_propagator_signal(returns, params.correlation_length_window)
    ssb_signal = _symmetry_breaking_signal(returns, prices, params.ssb_detection_window)

    field_signal = float(0.5 * prop_signal + 0.5 * ssb_signal)
    field_signal = float(np.clip(field_signal, -1, 1))

    # ── Composite ─────────────────────────────────────────────────────────────
    composite = float(
        params.gravitational_weight * grav_signal
        + params.thermodynamic_weight * thermo_signal
        + params.field_theory_weight * field_signal
    )
    composite = float(np.clip(composite, -1, 1))

    # Regime classification
    if abs(composite) < 0.1:
        regime = "neutral"
    elif temp > temp_baseline * 2.5:
        regime = "high_energy_chaotic"
    elif spin > 0 and mass > returns[-20:].std():
        regime = "spinning_trend"
    elif entropy_sig == 0.0:
        regime = "maximum_entropy"
    elif abs(ssb_signal) > 0.5:
        regime = "symmetry_broken_directional"
    else:
        regime = "normal_oscillating"

    # Confidence: agreement between layers
    signals = [grav_signal, thermo_signal, field_signal]
    signs = [np.sign(s) for s in signals if abs(s) > 0.1]
    if len(signs) >= 2:
        agreement = sum(1 for s in signs if s == signs[0]) / len(signs)
        confidence = float(agreement * abs(composite))
    else:
        confidence = float(abs(composite) * 0.5)

    return BHSignalComponents(
        gravitational=float(grav_signal),
        thermodynamic=float(thermo_signal),
        field_theory=float(field_signal),
        composite=composite,
        regime=regime,
        confidence=confidence,
        warnings=warnings,
    )


def bh_composite_signal_series(
    prices: np.ndarray,
    volume: Optional[np.ndarray] = None,
    params: Optional[BHCompositeParams] = None,
    min_window: int = 60,
) -> np.ndarray:
    """
    Compute rolling BH composite signal over the full price series.
    Returns signal array of same length as prices.
    """
    if params is None:
        params = BHCompositeParams()

    T = len(prices)
    signal = np.zeros(T)

    for t in range(min_window, T):
        p_window = prices[max(0, t - params.ssb_detection_window): t + 1]
        result = compute_bh_composite(p_window, None, params)
        signal[t] = result.composite

    return signal
