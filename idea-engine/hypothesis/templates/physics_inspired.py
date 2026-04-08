"""
idea-engine/hypothesis/templates/physics_inspired.py

Physics-inspired trading hypothesis templates.

Six templates that map physics metaphors to market structure:
  1. entropy_minimization   — low entropy regime → trend bet
  2. critical_point_fade    — near critical point (high susceptibility) → fade
  3. ergosphere_extraction  — high-spin BH metric → momentum follow
  4. phase_transition_reversal — after phase transition → MR back to equilibrium
  5. holographic_capacity_breach — information capacity maxed → position reduce
  6. renormalization_flow   — scale-invariant signal → hold regardless of TF

Each template produces a hypothesis with entry_condition, direction,
sizing_rule, exit_condition, and physics_metaphor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# ── Enums and Types ───────────────────────────────────────────────────────────

class HypothesisDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    LONG_OR_SHORT = "long_or_short"   # direction determined by signal sign


class SizingRule(str, Enum):
    FIXED = "fixed"
    VOLATILITY_SCALED = "volatility_scaled"
    KELLY = "kelly"
    ENTROPY_SCALED = "entropy_scaled"
    SUSCEPTIBILITY_INVERSE = "susceptibility_inverse"   # size inversely to susceptibility
    ERGOSPHERE_WEIGHTED = "ergosphere_weighted"


class ExitCondition(str, Enum):
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    REGIME_CHANGE = "regime_change"
    ENTROPY_NORMALIZED = "entropy_normalized"
    PHASE_COMPLETE = "phase_complete"
    CAPACITY_RESTORED = "capacity_restored"
    SCALE_BREAK = "scale_break"


# ── Template Base ─────────────────────────────────────────────────────────────

@dataclass
class PhysicsTemplate:
    """A single physics-inspired trading hypothesis template."""
    name: str
    physics_metaphor: str
    entry_condition: str         # human-readable description
    direction: HypothesisDirection
    sizing_rule: SizingRule
    exit_condition: ExitCondition
    parameters: Dict[str, Any]

    # Computed at signal time
    signal_value: float = 0.0
    direction_sign: float = 0.0   # +1 = long, -1 = short, 0 = flat
    suggested_size: float = 0.0
    confidence: float = 0.0
    active: bool = False

    # Physics quantities
    physics_analog: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            f"Template: {self.name}",
            f"  Metaphor:  {self.physics_metaphor}",
            f"  Entry:     {self.entry_condition}",
            f"  Direction: {self.direction.value} (sign={self.direction_sign:+.2f})",
            f"  Sizing:    {self.sizing_rule.value}",
            f"  Exit:      {self.exit_condition.value}",
            f"  Signal:    {self.signal_value:.4f}",
            f"  Size:      {self.suggested_size:.3f}",
            f"  Confidence:{self.confidence:.2f}",
            f"  Active:    {self.active}",
        ]
        if self.physics_analog:
            for k, v in self.physics_analog.items():
                lines.append(f"  [{k}: {v:.4f}]")
        return "\n".join(lines)


# ── Physics Signal Utilities ──────────────────────────────────────────────────

def sample_entropy(series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Sample entropy (SampEn) as a measure of time series complexity.
    Low SampEn = regular/predictable = low entropy.
    r is tolerance as fraction of std.
    """
    n = len(series)
    if n < 2 * (m + 1):
        return float("nan")
    r_abs = r * series.std()

    def _count_matches(template_len):
        count = 0
        total = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(series[i:i+template_len] - series[j:j+template_len])) < r_abs:
                    count += 1
            total += n - template_len - i - 1
        return count, total

    A, total_A = _count_matches(m + 1)
    B, total_B = _count_matches(m)

    if B == 0 or total_B == 0:
        return float("nan")
    ratio = (A / max(total_A, 1)) / (B / max(total_B, 1))
    if ratio <= 0:
        return float("nan")
    return -math.log(ratio)


def hurst_dfa(series: np.ndarray) -> float:
    """
    Detrended Fluctuation Analysis for Hurst exponent.
    Returns H ∈ (0,1). H > 0.5 = persistent/trending, H < 0.5 = anti-persistent.
    """
    n = len(series)
    if n < 20:
        return 0.5
    cumdev = np.cumsum(series - series.mean())
    scales = np.floor(np.logspace(1, np.log10(n // 4), 15)).astype(int)
    scales = np.unique(scales[scales > 4])
    fluct = []
    for scale in scales:
        segments = n // scale
        if segments < 1:
            continue
        dfa_f = []
        for seg in range(segments):
            chunk = cumdev[seg * scale:(seg + 1) * scale]
            x = np.arange(len(chunk))
            fit = np.polyfit(x, chunk, 1)
            trend = np.polyval(fit, x)
            dfa_f.append(np.sqrt(np.mean((chunk - trend)**2)))
        fluct.append(np.mean(dfa_f))

    if len(fluct) < 2:
        return 0.5
    slope, _, _, _, _ = sp_stats.linregress(np.log(scales[:len(fluct)]), np.log(fluct))
    return float(np.clip(slope, 0.01, 0.99))


def magnetic_susceptibility(returns: np.ndarray, window: int = 21) -> float:
    """
    Market analogy to magnetic susceptibility: d<m>/dh.
    Here: sensitivity of mean return to small "field" (lagged return).
    High susceptibility = near critical point = fragile market.
    Estimated as |slope| of regression of returns on lagged returns.
    """
    n = len(returns)
    if n < window + 2:
        return 0.0
    y = returns[-window:]
    x = returns[-window-1:-1]
    if x.std() < 1e-10:
        return 0.0
    slope, _, _, _, _ = sp_stats.linregress(x, y)
    return abs(float(slope))


def kerr_spin_metric(returns: np.ndarray, window: int = 21) -> float:
    """
    Analogy to Kerr black hole spin parameter a/M.
    Measures angular momentum of price motion = ratio of
    signed trend magnitude to total variation (like spin to mass ratio).
    High spin = strong directional momentum = ergosphere present.
    Range: [-1, 1]
    """
    n = len(returns)
    if n < window:
        return 0.0
    r = returns[-window:]
    signed_sum = r.sum()
    total_var = np.abs(r).sum()
    if total_var < 1e-10:
        return 0.0
    return float(signed_sum / total_var)


def order_parameter(returns: np.ndarray, window: int = 30) -> float:
    """
    Statistical mechanics order parameter: measures degree of ordering.
    Analogous to magnetization M = <sigma> in Ising model.
    Positive = bullish ordering, negative = bearish, near zero = disordered.
    """
    if len(returns) < window:
        return 0.0
    r = returns[-window:]
    signs = np.sign(r)
    return float(signs.mean())


def holographic_information_load(
    returns: np.ndarray,
    volume: Optional[np.ndarray] = None,
    window: int = 21,
) -> float:
    """
    Holographic principle analog: information encoded on the boundary
    bounds the bulk information content.
    Proxy: ratio of realized variance to historical average = info load.
    > 1.5 = near capacity, > 2.0 = capacity breach.
    """
    if len(returns) < window * 2:
        return 1.0
    rv_recent = np.var(returns[-window:])
    rv_hist = np.var(returns[-window * 2:-window])
    load = rv_recent / max(rv_hist, 1e-12)
    if volume is not None and len(volume) >= window * 2:
        vol_ratio = volume[-window:].mean() / max(volume[-window*2:-window].mean(), 1e-10)
        load = (load + vol_ratio) / 2.0
    return float(load)


def scale_invariance_score(returns: np.ndarray) -> float:
    """
    Test for scale invariance (renormalization group fixed point).
    Compare autocorrelation structure across time scales.
    Score near 1.0 = scale-invariant signal (hold regardless of TF).
    Score near 0.0 = scale-dependent (TF matters).
    Uses Hurst exponent proximity to 0.5 as proxy for scale-free behavior.
    Actually in RG sense, scale-invariant means the signal looks the same
    at different scales — we approximate this by comparing Hurst at multiple scales.
    """
    n = len(returns)
    if n < 64:
        return 0.5
    H_full = hurst_dfa(returns)
    H_half = hurst_dfa(returns[-n//2:])
    H_quarter = hurst_dfa(returns[-n//4:])
    # Scale invariance: Hurst should be consistent across scales
    h_vals = np.array([H_full, H_half, H_quarter])
    consistency = 1.0 - (h_vals.std() / max(h_vals.mean(), 0.01))
    return float(np.clip(consistency, 0.0, 1.0))


# ── Template 1: Entropy Minimization ─────────────────────────────────────────

def entropy_minimization(
    returns: np.ndarray,
    entropy_threshold: float = 0.5,
    max_size: float = 1.0,
) -> PhysicsTemplate:
    """
    Low entropy regime → directional trend bet.

    Physics: systems evolve toward minimum entropy configurations.
    A market in a low-entropy state is organized, predictable, and trending.
    When SampEn is low + Hurst > 0.55 → trend exists → follow.

    Entry: SampEn < threshold AND H > 0.55
    Direction: sign of recent trend (Hurst direction)
    Sizing: proportional to (1 - normalized_entropy)
    Exit: entropy rises above 2x threshold (disorder returns)
    """
    sampen = sample_entropy(returns[-60:] if len(returns) >= 60 else returns)
    H = hurst_dfa(returns)
    trend_sign = np.sign(returns[-20:].sum()) if len(returns) >= 20 else 0.0

    sampen_valid = not math.isnan(sampen)
    entropy_low = sampen_valid and sampen < entropy_threshold
    hurst_trending = H > 0.55

    active = entropy_low and hurst_trending
    norm_entropy = min(sampen / entropy_threshold, 2.0) if sampen_valid else 1.0
    confidence = float(np.clip((entropy_threshold - (sampen if sampen_valid else entropy_threshold)) / entropy_threshold, 0.0, 1.0))
    size = max_size * confidence * (H - 0.5) / 0.5 if hurst_trending else 0.0
    size = float(np.clip(size, 0.0, max_size))

    direction = (
        HypothesisDirection.LONG if trend_sign > 0 else
        HypothesisDirection.SHORT if trend_sign < 0 else
        HypothesisDirection.FLAT
    )

    return PhysicsTemplate(
        name="entropy_minimization",
        physics_metaphor=(
            "Thermodynamic entropy: ordered systems (low entropy) evolve predictably. "
            "Low SampEn + persistent Hurst = the market is in an organized, low-entropy state "
            "where trend strategies have positive expected value."
        ),
        entry_condition=f"SampEn < {entropy_threshold:.2f} AND Hurst > 0.55",
        direction=direction,
        sizing_rule=SizingRule.ENTROPY_SCALED,
        exit_condition=ExitCondition.ENTROPY_NORMALIZED,
        parameters={"entropy_threshold": entropy_threshold, "max_size": max_size},
        signal_value=float(sampen) if sampen_valid else float("nan"),
        direction_sign=float(trend_sign),
        suggested_size=size,
        confidence=confidence,
        active=active,
        physics_analog={
            "sample_entropy": float(sampen) if sampen_valid else float("nan"),
            "hurst_exponent": float(H),
            "trend_sign": float(trend_sign),
            "normalized_entropy": float(norm_entropy),
        },
    )


# ── Template 2: Critical Point Fade ──────────────────────────────────────────

def critical_point_fade(
    returns: np.ndarray,
    susceptibility_threshold: float = 0.4,
    max_size: float = 1.0,
) -> PhysicsTemplate:
    """
    Near critical point (high susceptibility) → fade the move.

    Physics: at a second-order phase transition, susceptibility χ = d<M>/dH → ∞.
    Markets near critical points show extreme sensitivity to order flow.
    High autocorrelation + high χ means price is fragile and overextended.

    Entry: susceptibility > threshold (system near phase transition)
    Direction: fade (opposite of recent move)
    Sizing: inversely proportional to susceptibility (smaller when more fragile)
    Exit: susceptibility drops below 0.5 * threshold (system relaxes)
    """
    chi = magnetic_susceptibility(returns)
    recent_return = float(returns[-5:].sum()) if len(returns) >= 5 else 0.0

    near_critical = chi > susceptibility_threshold
    fade_direction = -np.sign(recent_return) if recent_return != 0 else 0.0

    active = near_critical and abs(fade_direction) > 0
    confidence = float(np.clip((chi - susceptibility_threshold) / susceptibility_threshold, 0.0, 1.0))
    # Size inversely to susceptibility: more susceptible = smaller size
    size = max_size / (1.0 + chi / susceptibility_threshold) * confidence
    size = float(np.clip(size, 0.0, max_size))

    direction = (
        HypothesisDirection.LONG if fade_direction > 0 else
        HypothesisDirection.SHORT if fade_direction < 0 else
        HypothesisDirection.FLAT
    )

    return PhysicsTemplate(
        name="critical_point_fade",
        physics_metaphor=(
            "Statistical mechanics critical point: near a second-order phase transition, "
            "susceptibility χ diverges and small perturbations cause large responses. "
            "High market susceptibility = fragile, overextended → fade the trend as reversion is imminent."
        ),
        entry_condition=f"Susceptibility χ > {susceptibility_threshold:.2f} AND recent move exists",
        direction=direction,
        sizing_rule=SizingRule.SUSCEPTIBILITY_INVERSE,
        exit_condition=ExitCondition.REGIME_CHANGE,
        parameters={"susceptibility_threshold": susceptibility_threshold, "max_size": max_size},
        signal_value=chi,
        direction_sign=float(fade_direction),
        suggested_size=size,
        confidence=confidence,
        active=active,
        physics_analog={
            "susceptibility": chi,
            "recent_return": recent_return,
            "fade_sign": float(fade_direction),
            "critical_ratio": chi / max(susceptibility_threshold, 1e-10),
        },
    )


# ── Template 3: Ergosphere Extraction ────────────────────────────────────────

def ergosphere_extraction(
    returns: np.ndarray,
    spin_threshold: float = 0.6,
    max_size: float = 1.0,
) -> PhysicsTemplate:
    """
    High-spin Kerr black hole metric → momentum follow.

    Physics: the ergosphere of a rotating (Kerr) black hole is a region where
    no static observer can exist — all objects are dragged along by frame-dragging.
    High-spin market = strong momentum, everything is dragged in one direction.
    The Penrose process extracts energy from the ergosphere.

    Entry: |spin| > threshold (strong momentum, frame-dragging in effect)
    Direction: follow the spin (momentum trade)
    Sizing: proportional to |spin| relative to max (ergosphere depth)
    Exit: spin drops below threshold / 2 (ergosphere shrinks)
    """
    spin = kerr_spin_metric(returns)
    abs_spin = abs(spin)

    in_ergosphere = abs_spin > spin_threshold
    active = in_ergosphere
    confidence = float(np.clip((abs_spin - spin_threshold) / (1.0 - spin_threshold), 0.0, 1.0))
    size = max_size * (abs_spin - spin_threshold) / (1.0 - spin_threshold) if in_ergosphere else 0.0
    size = float(np.clip(size, 0.0, max_size))

    direction = (
        HypothesisDirection.LONG if spin > 0 else
        HypothesisDirection.SHORT if spin < 0 else
        HypothesisDirection.FLAT
    )

    return PhysicsTemplate(
        name="ergosphere_extraction",
        physics_metaphor=(
            "Kerr black hole ergosphere: in the rotating ergosphere, frame-dragging forces "
            "all objects to co-rotate. A high-spin market has strong momentum that drags "
            "all participants in one direction. The Penrose process: extract energy by "
            "riding the rotational momentum before it dissipates."
        ),
        entry_condition=f"|Kerr spin a/M| > {spin_threshold:.2f}",
        direction=direction,
        sizing_rule=SizingRule.ERGOSPHERE_WEIGHTED,
        exit_condition=ExitCondition.REGIME_CHANGE,
        parameters={"spin_threshold": spin_threshold, "max_size": max_size},
        signal_value=spin,
        direction_sign=float(np.sign(spin)),
        suggested_size=size,
        confidence=confidence,
        active=active,
        physics_analog={
            "kerr_spin": spin,
            "ergosphere_depth": float(max(abs_spin - spin_threshold, 0.0)),
            "frame_dragging_strength": abs_spin,
        },
    )


# ── Template 4: Phase Transition Reversal ────────────────────────────────────

def phase_transition_reversal(
    returns: np.ndarray,
    volume: Optional[np.ndarray] = None,
    order_threshold: float = 0.5,
    max_size: float = 1.0,
) -> PhysicsTemplate:
    """
    After a phase transition → mean reversion back to equilibrium.

    Physics: after a first-order phase transition (e.g., liquid→gas), the system
    is out of equilibrium and relaxes toward the new equilibrium state.
    Market analog: after a volatility spike or liquidity crisis (phase transition),
    the system overshoots and reverts. Order parameter goes from |M|=1 to |M|=0.

    Entry: |order parameter| was recently high, now falling (transition complete)
    Direction: opposite of the transition direction (MR)
    Sizing: proportional to how far from equilibrium (|order parameter| deviation)
    Exit: order parameter normalized to near-zero
    """
    M = order_parameter(returns)
    M_prev = order_parameter(returns[:-5]) if len(returns) > 25 else M
    transition_occurred = abs(M_prev) > order_threshold and abs(M) < order_threshold * 0.7

    # Reversion direction: away from the recent ordering
    reversion_sign = -np.sign(M_prev) if transition_occurred else 0.0
    active = transition_occurred and abs(reversion_sign) > 0

    deviation = float(abs(M_prev) - abs(M))
    confidence = float(np.clip(deviation / order_threshold, 0.0, 1.0))
    size = max_size * confidence * abs(M_prev) if active else 0.0
    size = float(np.clip(size, 0.0, max_size))

    direction = (
        HypothesisDirection.LONG if reversion_sign > 0 else
        HypothesisDirection.SHORT if reversion_sign < 0 else
        HypothesisDirection.FLAT
    )

    return PhysicsTemplate(
        name="phase_transition_reversal",
        physics_metaphor=(
            "First-order phase transition: when a system transitions between phases, "
            "it passes through a metastable state before settling. Market phase transitions "
            "(vol spikes, liquidity crises) create temporary disequilibrium. "
            "Post-transition, the system relaxes back to equilibrium — fade the extreme."
        ),
        entry_condition=(
            f"|order_parameter_prev| > {order_threshold:.2f} AND "
            f"|order_parameter_now| < {order_threshold * 0.7:.2f} (transition detected)"
        ),
        direction=direction,
        sizing_rule=SizingRule.VOLATILITY_SCALED,
        exit_condition=ExitCondition.PHASE_COMPLETE,
        parameters={"order_threshold": order_threshold, "max_size": max_size},
        signal_value=float(M),
        direction_sign=float(reversion_sign),
        suggested_size=size,
        confidence=confidence,
        active=active,
        physics_analog={
            "order_parameter_now": float(M),
            "order_parameter_prev": float(M_prev),
            "transition_magnitude": float(abs(M_prev) - abs(M)),
            "equilibrium_deviation": float(abs(M)),
        },
    )


# ── Template 5: Holographic Capacity Breach ──────────────────────────────────

def holographic_capacity_breach(
    returns: np.ndarray,
    volume: Optional[np.ndarray] = None,
    capacity_threshold: float = 1.8,
    max_size: float = 1.0,
) -> PhysicsTemplate:
    """
    When information capacity is maxed → reduce position size.

    Physics: holographic principle (Bekenstein bound) — maximum information
    that can be stored in a region is proportional to its boundary area.
    When a market is processing information at maximum capacity (high vol,
    high volume, many signals), the risk of model failure increases.
    Signal: reduce size proportionally to how near capacity we are.

    Entry: info_load > threshold (capacity stressed)
    Direction: FLAT (position reduction, not directional)
    Sizing: inversely proportional to info load (more load = smaller)
    Exit: info load drops below threshold (capacity restored)
    """
    load = holographic_information_load(returns, volume)
    capacity_breached = load > capacity_threshold

    active = capacity_breached
    overload = float(max(load / capacity_threshold - 1.0, 0.0))
    confidence = float(np.clip(overload, 0.0, 1.0))
    # Sizing: reduce position by load factor
    size_reduction = float(np.clip(load / capacity_threshold, 1.0, 3.0))
    suggested_size = max_size / size_reduction

    return PhysicsTemplate(
        name="holographic_capacity_breach",
        physics_metaphor=(
            "Bekenstein-Hawking holographic bound: the maximum information stored "
            "in a system is bounded by its boundary area (S ≤ A/4 in Planck units). "
            "When market information capacity is saturated (high vol + high volume = "
            "dense information density), model reliability degrades — reduce exposure."
        ),
        entry_condition=f"info_load > {capacity_threshold:.1f} (realized vol ratio vs history)",
        direction=HypothesisDirection.FLAT,
        sizing_rule=SizingRule.SUSCEPTIBILITY_INVERSE,
        exit_condition=ExitCondition.CAPACITY_RESTORED,
        parameters={"capacity_threshold": capacity_threshold, "max_size": max_size},
        signal_value=float(load),
        direction_sign=0.0,
        suggested_size=float(suggested_size),
        confidence=confidence,
        active=active,
        physics_analog={
            "information_load": float(load),
            "capacity_ratio": float(load / capacity_threshold),
            "overload": float(overload),
            "size_reduction_factor": float(size_reduction),
        },
    )


# ── Template 6: Renormalization Flow ─────────────────────────────────────────

def renormalization_flow(
    returns: np.ndarray,
    scale_invariance_threshold: float = 0.75,
    max_size: float = 1.0,
) -> PhysicsTemplate:
    """
    Scale-invariant signal → hold the position regardless of timeframe.

    Physics: renormalization group (RG) flow describes how physical laws change
    with scale. At RG fixed points, physics is scale-invariant (same at all scales).
    A scale-invariant trading signal is equally valid on 1m, 1h, and 1d charts —
    this is unusually robust and warrants higher confidence.

    Entry: scale_invariance_score > threshold
    Direction: consistent Hurst direction (persistent trend across all scales)
    Sizing: confidence proportional to scale invariance score
    Exit: scale invariance breaks (scale-dependent behavior returns)
    """
    si_score = scale_invariance_score(returns)
    H = hurst_dfa(returns)
    trend_sign = np.sign(returns[-20:].sum()) if len(returns) >= 20 else 0.0

    scale_invariant = si_score > scale_invariance_threshold and H > 0.5
    active = scale_invariant

    confidence = float(np.clip(
        (si_score - scale_invariance_threshold) / (1.0 - scale_invariance_threshold),
        0.0, 1.0
    ))
    size = max_size * confidence * float(np.clip((H - 0.5) / 0.5, 0.0, 1.0))
    size = float(np.clip(size, 0.0, max_size))

    direction = (
        HypothesisDirection.LONG if trend_sign > 0 and H > 0.5 else
        HypothesisDirection.SHORT if trend_sign < 0 and H > 0.5 else
        HypothesisDirection.FLAT
    )

    return PhysicsTemplate(
        name="renormalization_flow",
        physics_metaphor=(
            "Renormalization group fixed points: at RG fixed points, physical observables "
            "are scale-invariant — the laws look the same at every length/energy scale. "
            "A scale-invariant market signal persists across timeframes (1m = 1h = 1d). "
            "This multi-scale coherence is rare and indicates a structural market force."
        ),
        entry_condition=(
            f"scale_invariance_score > {scale_invariance_threshold:.2f} AND Hurst > 0.50"
        ),
        direction=direction,
        sizing_rule=SizingRule.KELLY,
        exit_condition=ExitCondition.SCALE_BREAK,
        parameters={"scale_invariance_threshold": scale_invariance_threshold, "max_size": max_size},
        signal_value=si_score,
        direction_sign=float(trend_sign),
        suggested_size=size,
        confidence=confidence,
        active=active,
        physics_analog={
            "scale_invariance_score": float(si_score),
            "hurst_exponent": float(H),
            "rg_fixed_point_proximity": float(si_score),
            "multi_scale_coherence": float(si_score * (H - 0.5) * 2.0),
        },
    )


# ── Composite: Run All Templates ──────────────────────────────────────────────

def run_all_templates(
    returns: np.ndarray,
    volume: Optional[np.ndarray] = None,
    max_size: float = 1.0,
    params_override: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, PhysicsTemplate]:
    """
    Run all six physics-inspired templates on the provided return series.

    Parameters
    ----------
    returns : 1D array of returns (daily or higher frequency)
    volume : optional volume series (same length as returns)
    max_size : maximum position size (normalized)
    params_override : optional dict to override template-specific parameters

    Returns
    -------
    dict mapping template_name → PhysicsTemplate result
    """
    if params_override is None:
        params_override = {}

    p1 = params_override.get("entropy_minimization", {})
    p2 = params_override.get("critical_point_fade", {})
    p3 = params_override.get("ergosphere_extraction", {})
    p4 = params_override.get("phase_transition_reversal", {})
    p5 = params_override.get("holographic_capacity_breach", {})
    p6 = params_override.get("renormalization_flow", {})

    templates = {
        "entropy_minimization": entropy_minimization(
            returns,
            entropy_threshold=p1.get("entropy_threshold", 0.5),
            max_size=p1.get("max_size", max_size),
        ),
        "critical_point_fade": critical_point_fade(
            returns,
            susceptibility_threshold=p2.get("susceptibility_threshold", 0.4),
            max_size=p2.get("max_size", max_size),
        ),
        "ergosphere_extraction": ergosphere_extraction(
            returns,
            spin_threshold=p3.get("spin_threshold", 0.6),
            max_size=p3.get("max_size", max_size),
        ),
        "phase_transition_reversal": phase_transition_reversal(
            returns,
            volume=volume,
            order_threshold=p4.get("order_threshold", 0.5),
            max_size=p4.get("max_size", max_size),
        ),
        "holographic_capacity_breach": holographic_capacity_breach(
            returns,
            volume=volume,
            capacity_threshold=p5.get("capacity_threshold", 1.8),
            max_size=p5.get("max_size", max_size),
        ),
        "renormalization_flow": renormalization_flow(
            returns,
            scale_invariance_threshold=p6.get("scale_invariance_threshold", 0.75),
            max_size=p6.get("max_size", max_size),
        ),
    }
    return templates


def active_templates(results: Dict[str, PhysicsTemplate]) -> List[PhysicsTemplate]:
    """Return only active templates sorted by confidence descending."""
    active = [t for t in results.values() if t.active]
    return sorted(active, key=lambda t: t.confidence, reverse=True)


def aggregate_signal(
    results: Dict[str, PhysicsTemplate],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Aggregate all active template signals into a single directional score.

    Returns (net_signal, net_size):
        net_signal ∈ [-1, 1]: positive = net long, negative = net short
        net_size ∈ [0, 1]: weighted average size
    """
    if weights is None:
        weights = {name: 1.0 for name in results}

    total_weight = 0.0
    weighted_signal = 0.0
    weighted_size = 0.0

    for name, template in results.items():
        if not template.active:
            continue
        w = weights.get(name, 1.0)
        # FLAT templates (holographic) reduce aggregate size but don't add direction
        if template.direction == HypothesisDirection.FLAT:
            # Apply a size reduction penalty
            weighted_size -= w * (1.0 - template.suggested_size) * template.confidence
        else:
            weighted_signal += w * template.direction_sign * template.confidence
            weighted_size += w * template.suggested_size
            total_weight += w

    if total_weight > 0:
        net_signal = float(np.clip(weighted_signal / total_weight, -1.0, 1.0))
        net_size = float(np.clip(weighted_size / total_weight, 0.0, 1.0))
    else:
        net_signal = 0.0
        net_size = 0.0

    return net_signal, net_size
