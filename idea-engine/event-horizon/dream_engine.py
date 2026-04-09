"""
Dream Engine: generative imagination for the autonomous trading system.

During market close (nights, weekends), the system DREAMS:
  1. Generates synthetic market scenarios that have never happened but could,
     based on the physics it has learned (perturbed physical constants)
  2. Stress-tests all live signals against dream scenarios to find fragilities
  3. Pre-trains the RL agent on dream scenarios (experience without risk)
  4. Breeds new physics concepts from dreams that produced surprising outcomes

This is imagination for a trading system: proactive simulation rather than
reactive inference. The system prepares for what MIGHT happen, not just
what HAS happened.
"""

from __future__ import annotations
import math
import time
import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dream: a synthetic market scenario
# ---------------------------------------------------------------------------

@dataclass
class Dream:
    """A synthetic market scenario generated from perturbed physics."""
    dream_id: str
    name: str
    physics_basis: str            # which physics analogy was perturbed
    perturbation: str             # what was changed
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    volatility_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_sequence: List[str] = field(default_factory=list)
    duration_bars: int = 252
    intensity: float = 1.0        # how extreme (1.0=normal, 5.0=nightmare)
    surprise_score: float = 0.0   # how different from any historical period
    metadata: Dict = field(default_factory=dict)


@dataclass
class FragilityReport:
    """Report on a signal's vulnerability in dream scenarios."""
    signal_name: str
    n_dreams_tested: int
    survival_rate: float          # fraction of dreams where signal is profitable
    worst_dream: str              # which dream caused the most damage
    worst_drawdown: float         # max DD in worst dream
    fragility_index: float        # 0=robust, 1=extremely fragile
    structural_contradictions: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class DreamInsight:
    """A new insight discovered from dream analysis."""
    insight_id: str
    description: str
    discovered_in_dream: str
    physics_concept: str
    potential_signal: str
    surprise_level: float
    actionability: float


# ---------------------------------------------------------------------------
# Physics Perturber: vary the physical constants
# ---------------------------------------------------------------------------

class PhysicsPerturber:
    """
    Perturb the physical constants that define market dynamics.

    Each physics analogy has parameters (like gravitational constant,
    spring constant, diffusion rate). By varying these stochastically,
    we generate market scenarios that are structurally valid but
    historically unprecedented.
    """

    PERTURBATION_PROFILES = {
        "gravity_weakening": {
            "description": "Support/resistance levels lose gravitational pull (liquidity drain)",
            "vol_multiplier": 1.5,
            "trend_strength": 0.3,
            "mean_reversion_speed": -0.05,
            "jump_probability": 0.02,
        },
        "gravity_intensification": {
            "description": "All prices collapse toward a single attractor (flash crash)",
            "vol_multiplier": 3.0,
            "trend_strength": -0.5,
            "mean_reversion_speed": 0.3,
            "jump_probability": 0.1,
        },
        "entropy_maximization": {
            "description": "Market reaches maximum disorder (random walk, no patterns)",
            "vol_multiplier": 1.0,
            "trend_strength": 0.0,
            "mean_reversion_speed": 0.0,
            "jump_probability": 0.0,
        },
        "phase_transition_slow": {
            "description": "Gradual regime shift over 60 bars (boiling water analogy)",
            "vol_multiplier": 0.5,
            "trend_strength": 0.0,
            "mean_reversion_speed": 0.1,
            "jump_probability": 0.0,
            "transition_at_bar": 60,
            "post_transition_vol": 3.0,
        },
        "phase_transition_instant": {
            "description": "Sudden regime shift (earthquake analogy)",
            "vol_multiplier": 0.3,
            "trend_strength": 0.1,
            "mean_reversion_speed": 0.05,
            "jump_probability": 0.0,
            "transition_at_bar": 1,
            "post_transition_vol": 5.0,
            "post_transition_trend": -0.8,
        },
        "resonance_buildup": {
            "description": "Oscillations grow in amplitude until system breaks (bridge collapse)",
            "vol_multiplier": 0.5,
            "trend_strength": 0.0,
            "mean_reversion_speed": 0.15,
            "oscillation_growth_rate": 0.02,
        },
        "quantum_tunneling": {
            "description": "Price jumps through a barrier that should hold (quantum tunneling)",
            "vol_multiplier": 0.8,
            "trend_strength": 0.0,
            "mean_reversion_speed": 0.1,
            "jump_probability": 0.05,
            "jump_size": 0.08,  # 8% jump
        },
        "heat_death": {
            "description": "All volatility slowly drains away (heat death of the universe)",
            "vol_multiplier": 1.0,
            "vol_decay_rate": 0.005,
            "trend_strength": 0.0,
            "mean_reversion_speed": 0.0,
        },
        "big_bang": {
            "description": "Sudden creation of volatility from nothing (IPO, new market)",
            "vol_multiplier": 0.1,
            "vol_growth_rate": 0.01,
            "trend_strength": 0.5,
            "mean_reversion_speed": 0.0,
        },
        "dark_energy": {
            "description": "Accelerating expansion (bubble with accelerating price growth)",
            "vol_multiplier": 0.8,
            "trend_strength": 0.1,
            "trend_acceleration": 0.002,
            "mean_reversion_speed": -0.05,
        },
    }

    def __init__(self, base_vol: float = 0.02, seed: int = 42):
        self.base_vol = base_vol
        self.rng = np.random.default_rng(seed)

    def generate_dream(self, profile_name: str, duration: int = 252,
                        intensity: float = 1.0) -> Dream:
        """Generate a dream scenario from a perturbation profile."""
        profile = self.PERTURBATION_PROFILES.get(
            profile_name, self.PERTURBATION_PROFILES["entropy_maximization"]
        )

        returns = np.zeros(duration)
        vol_profile = np.full(duration, self.base_vol)

        vol_mult = profile.get("vol_multiplier", 1.0) * intensity
        trend = profile.get("trend_strength", 0.0) * intensity
        mr_speed = profile.get("mean_reversion_speed", 0.0)
        jump_prob = profile.get("jump_probability", 0.0) * intensity
        jump_size = profile.get("jump_size", 0.03) * intensity

        # Dynamic parameters
        vol_decay = profile.get("vol_decay_rate", 0.0)
        vol_growth = profile.get("vol_growth_rate", 0.0)
        trend_accel = profile.get("trend_acceleration", 0.0) * intensity
        osc_growth = profile.get("oscillation_growth_rate", 0.0) * intensity
        transition_bar = profile.get("transition_at_bar", duration + 1)
        post_vol = profile.get("post_transition_vol", vol_mult) * intensity
        post_trend = profile.get("post_transition_trend", trend)

        current_vol = self.base_vol * vol_mult
        current_trend = trend / 252

        for t in range(duration):
            # Phase transition check
            if t == transition_bar:
                current_vol = self.base_vol * post_vol
                current_trend = post_trend / 252

            # Vol dynamics
            if vol_decay > 0:
                current_vol *= (1 - vol_decay)
            if vol_growth > 0:
                current_vol *= (1 + vol_growth)
            current_vol = max(current_vol, 1e-6)

            # Trend dynamics
            if trend_accel != 0:
                current_trend += trend_accel / 252

            # Oscillation
            if osc_growth > 0:
                amplitude = osc_growth * t
                current_trend += amplitude * math.sin(2 * math.pi * t / 20) / 252

            # Base return
            ret = self.rng.normal(current_trend, current_vol / math.sqrt(252))

            # Mean reversion
            if t > 0 and mr_speed != 0:
                ret -= mr_speed * returns[t - 1]

            # Jumps
            if self.rng.random() < jump_prob / 252:
                ret += self.rng.normal(0, jump_size)

            returns[t] = ret
            vol_profile[t] = current_vol

        # Regime labels
        regimes = []
        for t in range(duration):
            v = vol_profile[t]
            if v > self.base_vol * 2:
                regimes.append("crisis")
            elif v > self.base_vol * 1.5:
                regimes.append("volatile")
            elif abs(returns[max(0, t-20):t+1].mean()) * 252 > 0.1:
                regimes.append("trending")
            else:
                regimes.append("normal")

        # Surprise score: KL divergence from standard normal
        if returns.std() > 1e-10:
            z = (returns - returns.mean()) / returns.std()
            surprise = float(abs(z).mean() - 0.798)  # E[|Z|] = sqrt(2/pi) ~ 0.798
        else:
            surprise = 0.0

        return Dream(
            dream_id=f"dream_{profile_name}_{int(time.time()) % 10000}",
            name=profile["description"][:80],
            physics_basis=profile_name,
            perturbation=profile["description"],
            returns=returns,
            volatility_profile=vol_profile,
            regime_sequence=regimes,
            duration_bars=duration,
            intensity=intensity,
            surprise_score=abs(surprise),
        )

    def generate_nightmare_suite(self, n_per_profile: int = 3,
                                   duration: int = 252) -> List[Dream]:
        """Generate a full suite of dream scenarios across all profiles."""
        dreams = []
        for profile_name in self.PERTURBATION_PROFILES:
            for i in range(n_per_profile):
                intensity = 1.0 + i * 0.5  # 1.0, 1.5, 2.0
                dream = self.generate_dream(profile_name, duration, intensity)
                dreams.append(dream)
        return dreams


# ---------------------------------------------------------------------------
# Fragility Tester: test signals against dreams
# ---------------------------------------------------------------------------

class FragilityTester:
    """
    Test live signals against dream scenarios to find vulnerabilities
    before they manifest in real markets.
    """

    def __init__(self, transaction_cost: float = 0.001):
        self.tc = transaction_cost

    def test_signal(
        self,
        signal_fn: Callable[[np.ndarray], np.ndarray],
        signal_name: str,
        dreams: List[Dream],
    ) -> FragilityReport:
        """Test a signal function against all dream scenarios."""
        results = []

        for dream in dreams:
            try:
                signal = signal_fn(dream.returns)
                strat_returns = signal[:-1] * dream.returns[1:]
                costs = np.abs(np.diff(signal, prepend=0))[:-1] * self.tc
                net = strat_returns - costs

                if len(net) > 10 and net.std() > 1e-10:
                    sharpe = float(net.mean() / net.std() * math.sqrt(252))
                    eq = np.cumprod(1 + net)
                    peak = np.maximum.accumulate(eq)
                    max_dd = float(((peak - eq) / peak).max())
                else:
                    sharpe = 0.0
                    max_dd = 0.0

                results.append({
                    "dream": dream.dream_id,
                    "dream_name": dream.name,
                    "sharpe": sharpe,
                    "max_dd": max_dd,
                    "profitable": sharpe > 0,
                    "physics_basis": dream.physics_basis,
                })
            except Exception:
                results.append({
                    "dream": dream.dream_id,
                    "dream_name": dream.name,
                    "sharpe": 0.0,
                    "max_dd": 1.0,
                    "profitable": False,
                    "physics_basis": dream.physics_basis,
                })

        if not results:
            return FragilityReport(signal_name, 0, 0.0, "", 0.0, 1.0)

        survival = sum(1 for r in results if r["profitable"]) / len(results)
        worst = min(results, key=lambda r: r["sharpe"])
        fragility = 1.0 - survival

        # Structural contradictions: physics concepts where the signal fails
        failing_physics = defaultdict(int)
        for r in results:
            if not r["profitable"]:
                failing_physics[r["physics_basis"]] += 1
        contradictions = [
            f"Fails in {physics} scenarios ({count} times)"
            for physics, count in sorted(failing_physics.items(), key=lambda x: -x[1])[:3]
        ]

        if fragility > 0.7:
            rec = "CRITICAL: Signal is highly fragile. Consider retirement or hedging."
        elif fragility > 0.4:
            rec = "WARNING: Signal has significant vulnerabilities. Add regime gating."
        else:
            rec = "HEALTHY: Signal survives most dream scenarios."

        return FragilityReport(
            signal_name=signal_name,
            n_dreams_tested=len(results),
            survival_rate=survival,
            worst_dream=worst["dream_name"],
            worst_drawdown=worst["max_dd"],
            fragility_index=fragility,
            structural_contradictions=contradictions,
            recommendation=rec,
        )


# ---------------------------------------------------------------------------
# Dream Insight Generator: discover new concepts from surprising dreams
# ---------------------------------------------------------------------------

class DreamInsightGenerator:
    """
    Analyze dream outcomes to discover new physics concepts.

    When a dream produces surprising results (signal that normally fails
    suddenly works, or vice versa), the generator extracts the structural
    features of that dream and proposes them as new signal primitives.
    """

    def __init__(self):
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"insight_{self._counter:04d}"

    def analyze_dreams(
        self,
        dreams: List[Dream],
        signal_results: Dict[str, List[Dict]],  # signal_name -> [{dream_id, sharpe, ...}]
    ) -> List[DreamInsight]:
        """
        Find surprising outcomes and generate new physics concepts.

        Surprising = signal that normally works suddenly fails, or
        signal that normally fails suddenly works. The dream conditions
        that cause this surprise contain new information.
        """
        insights = []

        for signal_name, results in signal_results.items():
            if len(results) < 5:
                continue

            sharpes = [r.get("sharpe", 0) for r in results]
            mean_sharpe = float(np.mean(sharpes))
            std_sharpe = float(np.std(sharpes))

            for r in results:
                sharpe = r.get("sharpe", 0)
                if std_sharpe < 1e-10:
                    continue

                z = (sharpe - mean_sharpe) / std_sharpe

                # Surprisingly good (signal works in dream where it shouldn't)
                if z > 2.0 and mean_sharpe < 0.5:
                    dream_id = r.get("dream", "")
                    dream = next((d for d in dreams if d.dream_id == dream_id), None)
                    if dream:
                        insights.append(DreamInsight(
                            insight_id=self._next_id(),
                            description=f"{signal_name} works surprisingly well in {dream.physics_basis} dream",
                            discovered_in_dream=dream.dream_id,
                            physics_concept=f"Hidden robustness in {dream.physics_basis} conditions",
                            potential_signal=f"Activate {signal_name} specifically during {dream.physics_basis} regimes",
                            surprise_level=float(z),
                            actionability=0.7,
                        ))

                # Surprisingly bad (signal fails in dream where it should work)
                elif z < -2.0 and mean_sharpe > 0.3:
                    dream_id = r.get("dream", "")
                    dream = next((d for d in dreams if d.dream_id == dream_id), None)
                    if dream:
                        insights.append(DreamInsight(
                            insight_id=self._next_id(),
                            description=f"{signal_name} fails catastrophically in {dream.physics_basis} dream",
                            discovered_in_dream=dream.dream_id,
                            physics_concept=f"Hidden fragility: {dream.physics_basis} breaks {signal_name}",
                            potential_signal=f"Hedge {signal_name} when {dream.physics_basis} conditions detected",
                            surprise_level=float(abs(z)),
                            actionability=0.9,
                        ))

        insights.sort(key=lambda i: i.surprise_level, reverse=True)
        return insights


# ---------------------------------------------------------------------------
# Dream Engine: the master orchestrator
# ---------------------------------------------------------------------------

class DreamEngine:
    """
    The imagination system for autonomous trading.

    During market close:
      1. Generate dream scenarios from perturbed physics
      2. Test all live signals against dreams
      3. Generate fragility reports
      4. Discover new insights from surprising outcomes
      5. Feed insights back to the EHS for new signal synthesis

    The system IMAGINES market scenarios and PREPARES for them
    before they happen.
    """

    def __init__(self, base_vol: float = 0.02, seed: int = 42):
        self.perturber = PhysicsPerturber(base_vol, seed)
        self.fragility = FragilityTester()
        self.insight_gen = DreamInsightGenerator()
        self._dream_history: List[Dream] = []
        self._report_history: List[Dict] = []

    def dream_session(
        self,
        signal_functions: Dict[str, Callable],
        n_per_profile: int = 2,
        duration: int = 126,
    ) -> Dict:
        """
        Run a full dreaming session.

        signal_functions: dict of signal_name -> function(returns) -> signal_array
        """
        session_start = time.time()

        # 1. Generate dreams
        dreams = self.perturber.generate_nightmare_suite(n_per_profile, duration)
        self._dream_history.extend(dreams)

        # 2. Test all signals
        fragility_reports = {}
        signal_results = {}

        for name, fn in signal_functions.items():
            report = self.fragility.test_signal(fn, name, dreams)
            fragility_reports[name] = report

            # Collect per-dream results for insight generation
            results = []
            for dream in dreams:
                try:
                    signal = fn(dream.returns)
                    strat = signal[:-1] * dream.returns[1:]
                    sharpe = float(strat.mean() / max(strat.std(), 1e-10) * math.sqrt(252)) if len(strat) > 10 else 0
                    results.append({"dream": dream.dream_id, "sharpe": sharpe, "physics_basis": dream.physics_basis})
                except:
                    results.append({"dream": dream.dream_id, "sharpe": 0, "physics_basis": dream.physics_basis})
            signal_results[name] = results

        # 3. Generate insights
        insights = self.insight_gen.analyze_dreams(dreams, signal_results)

        # 4. Compile report
        report = {
            "session_time": time.time() - session_start,
            "n_dreams": len(dreams),
            "n_signals_tested": len(signal_functions),
            "fragility_reports": {
                name: {
                    "survival_rate": r.survival_rate,
                    "fragility_index": r.fragility_index,
                    "worst_dream": r.worst_dream,
                    "worst_drawdown": r.worst_drawdown,
                    "recommendation": r.recommendation,
                    "contradictions": r.structural_contradictions,
                }
                for name, r in fragility_reports.items()
            },
            "insights": [
                {
                    "id": i.insight_id,
                    "description": i.description,
                    "physics": i.physics_concept,
                    "potential_signal": i.potential_signal,
                    "surprise": i.surprise_level,
                }
                for i in insights[:10]
            ],
            "most_fragile_signal": min(fragility_reports.values(), key=lambda r: r.survival_rate).signal_name if fragility_reports else None,
            "most_robust_signal": max(fragility_reports.values(), key=lambda r: r.survival_rate).signal_name if fragility_reports else None,
        }

        self._report_history.append(report)
        return report
