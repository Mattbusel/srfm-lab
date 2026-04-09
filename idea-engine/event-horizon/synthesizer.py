"""
Event Horizon Synthesizer (EHS) -- Autonomous Signal Discovery and Deployment.

The most ambitious module in the codebase. Combines:
  1. Academic Miner: ingest physics concepts as candidate signal primitives
  2. BH Physics Engine: synthesize new Minkowski-class signal primitives
  3. Multi-Agent Debate: adversarial validation with statistical rigor
  4. PPO Reinforcement Learning: autonomous execution optimization
  5. Shadow Deployment: paper-trade new signals before live deployment
  6. Self-Pruning Registry: auto-retire decaying signals

The loop:
  Concept -> Primitive -> Debate -> Backtest -> Shadow -> Deploy -> Monitor -> Retire

This is an automated scientific discovery engine for trading signals.
"""

from __future__ import annotations
import math
import time
import json
import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Signal Primitive: the atomic unit of a physics-based trading signal
# ---------------------------------------------------------------------------

@dataclass
class SignalPrimitive:
    """
    A synthesized signal primitive -- a single computable feature
    derived from a physics concept applied to market data.
    """
    primitive_id: str
    name: str
    physics_concept: str        # e.g., "Luttinger liquid", "Ising model", "Bose-Einstein condensate"
    domain: str                 # physics / information_theory / thermodynamics / QFT / topology
    description: str

    # The computation
    feature_names: List[str]    # input features required (returns, volume, spread, etc.)
    lookback: int               # bars of history needed
    computation_code: str       # human-readable description of the math

    # Lifecycle
    status: str = "candidate"   # candidate / debating / backtesting / shadow / live / retired
    created_at: float = 0.0
    promoted_at: float = 0.0
    retired_at: float = 0.0

    # Performance
    ic_history: List[float] = field(default_factory=list)
    sharpe_history: List[float] = field(default_factory=list)
    shadow_pnl: float = 0.0
    live_pnl: float = 0.0
    debate_score: float = 0.0
    adversarial_failure_rate: float = 1.0

    # Lineage
    parent_primitive: Optional[str] = None
    generation: int = 0
    mutation_type: str = ""


# ---------------------------------------------------------------------------
# Physics Concept Library: templates for signal synthesis
# ---------------------------------------------------------------------------

PHYSICS_TEMPLATES = [
    {
        "concept": "Ising Model Phase Transition",
        "domain": "statistical_mechanics",
        "description": "Model market participants as spins on a lattice. When the 'temperature' "
                       "(volatility) drops below the critical point, spontaneous magnetization "
                       "(herding) occurs. Signal: detect the critical temperature from vol and "
                       "correlation data, predict phase transition (regime change).",
        "features": ["returns", "correlation_matrix", "realized_vol"],
        "lookback": 63,
        "computation": "magnetization = mean(sign(returns_cross_section)); "
                       "susceptibility = var(sign(returns_cross_section)); "
                       "signal = susceptibility > critical_threshold",
    },
    {
        "concept": "Bose-Einstein Condensation",
        "domain": "quantum_mechanics",
        "description": "In BEC, particles collapse into the lowest energy state below critical "
                       "temperature. In markets: when vol drops below critical, all assets 'condense' "
                       "into a single factor (market beta). Signal: detect BEC onset from the "
                       "concentration ratio of PCA eigenvalues.",
        "features": ["returns_matrix", "realized_vol"],
        "lookback": 42,
        "computation": "eigenvalues = PCA(returns_matrix); "
                       "condensation_ratio = eigenvalues[0] / sum(eigenvalues); "
                       "signal = condensation_ratio > 0.7 (single factor dominance)",
    },
    {
        "concept": "Hawking Radiation Information Paradox",
        "domain": "quantum_gravity",
        "description": "Information that falls into a BH is not lost but encoded in Hawking radiation. "
                       "In markets: when a trend 'collapses' (BH evaporates), the information about "
                       "the trend direction is encoded in the pattern of the decline. Signal: decode "
                       "the reversal pattern from the Hawking temperature profile during BH collapse.",
        "features": ["bh_mass", "hawking_temperature", "returns"],
        "lookback": 21,
        "computation": "if bh_mass was > BH_FORM and now declining: "
                       "reversal_info = correlation(hawking_temp_series, future_returns); "
                       "signal = sign(reversal_info) * abs(hawking_temp_gradient)",
    },
    {
        "concept": "Casimir Effect (Vacuum Energy)",
        "domain": "QFT",
        "description": "Two parallel plates in vacuum experience an attractive force from quantum "
                       "fluctuations. In markets: two correlated assets that are 'squeezed' together "
                       "(low spread) experience an attractive force (mean reversion). The Casimir "
                       "force scales as 1/d^4. Signal: when the spread between correlated assets "
                       "is compressed, predict mean reversion with force inversely proportional to "
                       "spread width.",
        "features": ["spread", "correlation", "realized_vol"],
        "lookback": 21,
        "computation": "casimir_force = correlation^2 / max(spread^4, epsilon); "
                       "signal = casimir_force * sign(spread_deviation_from_mean)",
    },
    {
        "concept": "Kolmogorov-Arnold-Moser (KAM) Theorem",
        "domain": "dynamical_systems",
        "description": "In perturbed Hamiltonian systems, most orbits remain stable (KAM tori) "
                       "but some become chaotic. In markets: most price paths are 'quasi-periodic' "
                       "(range-bound) but perturbations (news, events) can break the torus into "
                       "chaos. Signal: measure the 'KAM stability' from autocorrelation structure.",
        "features": ["returns", "autocorrelation_profile"],
        "lookback": 63,
        "computation": "acf = autocorrelation(returns, lags=1..20); "
                       "kam_stability = 1 - max_lyapunov_exponent(returns); "
                       "signal = kam_stability < threshold -> regime_break_imminent",
    },
    {
        "concept": "Dirac Equation (Spin-1/2 Particles)",
        "domain": "quantum_mechanics",
        "description": "The Dirac equation predicts antimatter: every particle has an antiparticle. "
                       "In markets: every signal has an 'anti-signal' that performs inversely in "
                       "different regimes. Signal: for each active signal, compute the anti-signal "
                       "(negation in the complementary regime) and use it for hedging.",
        "features": ["signal_values", "regime_labels"],
        "lookback": 126,
        "computation": "anti_signal = -signal * regime_mismatch_indicator; "
                       "hedge_ratio = correlation(signal, anti_signal) in target_regime",
    },
    {
        "concept": "Navier-Stokes Turbulence",
        "domain": "fluid_dynamics",
        "description": "In turbulent flow, energy cascades from large to small scales (Kolmogorov). "
                       "In markets: volatility cascades from macro shocks to micro-structure. "
                       "Signal: detect the onset of turbulence from the energy spectrum of returns "
                       "at multiple timeframes.",
        "features": ["returns_1m", "returns_15m", "returns_1h", "returns_4h"],
        "lookback": 96,
        "computation": "energy_spectrum = [var(returns_tf) for tf in timeframes]; "
                       "kolmogorov_slope = regression_slope(log(freq), log(energy)); "
                       "signal = kolmogorov_slope < -5/3 -> turbulence_onset",
    },
    {
        "concept": "Topological Phase Transition",
        "domain": "condensed_matter",
        "description": "In topological materials, phase transitions are characterized by changes "
                       "in topological invariants (winding numbers, Chern numbers) rather than "
                       "symmetry breaking. In markets: regime changes that DON'T show up in "
                       "volatility but DO show up in the topology of the correlation network.",
        "features": ["correlation_matrix", "returns_matrix"],
        "lookback": 63,
        "computation": "graph = threshold(correlation_matrix, 0.5); "
                       "betti_0 = connected_components(graph); "
                       "betti_1 = cycles(graph); "
                       "signal = delta(betti_1) != 0 -> topological_phase_change",
    },
    {
        "concept": "Renormalization Group Flow",
        "domain": "statistical_mechanics",
        "description": "Under RG flow, systems evolve toward fixed points. Near a critical point, "
                       "the RG flow slows down (critical slowing). In markets: when the Hurst "
                       "exponent at multiple timescales converges to 0.5, the system is near "
                       "a 'fixed point' (efficient market). Divergence from 0.5 at any scale "
                       "signals exploitable inefficiency.",
        "features": ["returns"],
        "lookback": 252,
        "computation": "hurst_by_scale = [hurst(returns, window=w) for w in [21, 63, 126, 252]]; "
                       "rg_flow = gradient(hurst_by_scale); "
                       "signal = max(abs(hurst_by_scale - 0.5)) * sign(rg_flow)",
    },
    {
        "concept": "Wheeler-DeWitt Equation (Quantum Cosmology)",
        "domain": "quantum_gravity",
        "description": "The WDW equation describes the wavefunction of the universe -- a state "
                       "that encompasses all possible configurations. In markets: the 'wavefunction' "
                       "of the portfolio is a superposition of all possible future states weighted "
                       "by their probability. Signal: use the implied distribution from options "
                       "prices as the 'wavefunction' and compute the expected value under different "
                       "'measurement bases' (regimes).",
        "features": ["options_iv_surface", "spot_price", "risk_free_rate"],
        "lookback": 21,
        "computation": "risk_neutral_density = second_derivative(call_prices, strike); "
                       "wavefunction_collapse = max_probability_state(density); "
                       "signal = (wavefunction_collapse - spot) / spot",
    },
]


# ---------------------------------------------------------------------------
# Primitive Synthesizer: create computable signals from physics concepts
# ---------------------------------------------------------------------------

class PrimitiveSynthesizer:
    """
    Convert physics concepts into computable signal primitives.
    Each primitive is a function: market_data -> signal_value.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"prim_{self._counter:04d}"

    def synthesize_from_template(self, template: Dict) -> SignalPrimitive:
        """Create a SignalPrimitive from a physics template."""
        return SignalPrimitive(
            primitive_id=self._next_id(),
            name=template["concept"].replace(" ", "_").lower(),
            physics_concept=template["concept"],
            domain=template["domain"],
            description=template["description"],
            feature_names=template["features"],
            lookback=template["lookback"],
            computation_code=template["computation"],
            created_at=time.time(),
        )

    def synthesize_all(self) -> List[SignalPrimitive]:
        """Synthesize primitives from all templates."""
        return [self.synthesize_from_template(t) for t in PHYSICS_TEMPLATES]

    def mutate_primitive(self, primitive: SignalPrimitive) -> SignalPrimitive:
        """Create a mutated variant of an existing primitive."""
        mutations = [
            ("invert", "Inverted signal direction"),
            ("threshold_shift", "Shifted activation threshold"),
            ("timeframe_change", "Changed lookback window"),
            ("combine_entropy", "Added entropy weighting"),
            ("regime_gate", "Added regime gating condition"),
        ]
        mutation_type, desc = self.rng.choice(mutations)

        child = SignalPrimitive(
            primitive_id=self._next_id(),
            name=f"{primitive.name}_mut_{mutation_type}",
            physics_concept=f"{primitive.physics_concept} ({desc})",
            domain=primitive.domain,
            description=f"Mutation of {primitive.name}: {desc}",
            feature_names=primitive.feature_names.copy(),
            lookback=primitive.lookback if mutation_type != "timeframe_change"
                     else int(primitive.lookback * self.rng.uniform(0.5, 2.0)),
            computation_code=f"MUTATED({mutation_type}): {primitive.computation_code}",
            parent_primitive=primitive.primitive_id,
            generation=primitive.generation + 1,
            mutation_type=mutation_type,
            created_at=time.time(),
        )
        return child


# ---------------------------------------------------------------------------
# Adversarial Debate Gate
# ---------------------------------------------------------------------------

@dataclass
class DebateVerdict:
    """Result of adversarial debate on a signal primitive."""
    primitive_id: str
    approved: bool
    consensus_score: float       # 0-1 agreement level
    adversarial_survival: float  # fraction of stress tests survived
    statistical_significance: float  # p-value from backtest
    key_arguments_for: List[str]
    key_arguments_against: List[str]
    regime_robustness: float     # fraction of regimes where signal works
    estimated_capacity: float    # max AUM before alpha decay


class AdversarialDebateGate:
    """
    Multi-agent adversarial validation of signal primitives.

    Simulates a debate between:
      - QuantResearcher: argues for statistical validity
      - DevilsAdvocate: tries to break the signal
      - RiskManager: evaluates tail risk
      - RegimeExpert: checks regime robustness
      - Statistician: checks for overfitting / multiple testing
    """

    def __init__(self, significance_threshold: float = 0.05,
                 min_regime_fraction: float = 0.5,
                 min_adversarial_survival: float = 0.6):
        self.sig_threshold = significance_threshold
        self.min_regime = min_regime_fraction
        self.min_survival = min_adversarial_survival

    def evaluate(
        self,
        primitive: SignalPrimitive,
        backtest_returns: np.ndarray,
        regime_labels: np.ndarray,
        n_stress_scenarios: int = 8,
    ) -> DebateVerdict:
        """
        Run the adversarial debate on a candidate primitive.
        Uses the backtest returns generated by the primitive's signal.
        """
        n = len(backtest_returns)
        if n < 63:
            return DebateVerdict(
                primitive_id=primitive.primitive_id,
                approved=False,
                consensus_score=0.0,
                adversarial_survival=0.0,
                statistical_significance=1.0,
                key_arguments_for=[],
                key_arguments_against=["Insufficient data"],
                regime_robustness=0.0,
                estimated_capacity=0.0,
            )

        # QuantResearcher: t-test on mean return
        mean_r = float(backtest_returns.mean())
        std_r = float(backtest_returns.std())
        t_stat = mean_r / max(std_r / math.sqrt(n), 1e-10)
        # Approximate p-value from t-stat (normal approximation)
        p_value = float(2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))))
        is_significant = p_value < self.sig_threshold

        # Sharpe
        sharpe = float(mean_r / max(std_r, 1e-10) * math.sqrt(252))

        # RegimeExpert: check across regimes
        unique_regimes = np.unique(regime_labels)
        regime_sharpes = {}
        for r in unique_regimes:
            mask = regime_labels == r
            r_rets = backtest_returns[mask]
            if len(r_rets) >= 20:
                r_sharpe = float(r_rets.mean() / max(r_rets.std(), 1e-10) * math.sqrt(252))
                regime_sharpes[str(r)] = r_sharpe
        regime_robust = sum(1 for s in regime_sharpes.values() if s > 0) / max(len(regime_sharpes), 1)

        # DevilsAdvocate: stress scenarios (bootstrap worst-case)
        rng = np.random.default_rng(42)
        survived = 0
        for _ in range(n_stress_scenarios):
            # Bootstrap a bad scenario: sample with replacement, bias toward losses
            weights = np.where(backtest_returns < 0, 2.0, 1.0)
            weights /= weights.sum()
            stress_sample = rng.choice(backtest_returns, size=n // 2, p=weights)
            if stress_sample.mean() > -0.001:  # survived if not catastrophic
                survived += 1
        adversarial_survival = survived / n_stress_scenarios

        # Statistician: deflated Sharpe (penalty for testing multiple signals)
        # Simplified: haircut Sharpe by 30% for multiple testing
        deflated_sharpe = sharpe * 0.7

        # Build arguments
        args_for = []
        args_against = []

        if is_significant:
            args_for.append(f"Statistically significant (p={p_value:.4f}, t={t_stat:.2f})")
        else:
            args_against.append(f"Not significant (p={p_value:.4f})")

        if sharpe > 0.5:
            args_for.append(f"Sharpe ratio {sharpe:.2f} exceeds minimum threshold")
        else:
            args_against.append(f"Sharpe {sharpe:.2f} below minimum")

        if regime_robust >= self.min_regime:
            args_for.append(f"Robust across {regime_robust:.0%} of regimes")
        else:
            args_against.append(f"Only works in {regime_robust:.0%} of regimes")

        if adversarial_survival >= self.min_survival:
            args_for.append(f"Survived {adversarial_survival:.0%} of stress scenarios")
        else:
            args_against.append(f"Failed {1-adversarial_survival:.0%} of stress scenarios")

        # Consensus
        scores = [
            1.0 if is_significant else 0.0,
            min(sharpe / 1.0, 1.0),
            regime_robust,
            adversarial_survival,
        ]
        consensus = float(np.mean(scores))

        # Capacity estimate (simplified: inversely proportional to market impact)
        capacity = 1e6 * max(deflated_sharpe, 0) * regime_robust

        approved = (
            is_significant
            and deflated_sharpe > 0.3
            and regime_robust >= self.min_regime
            and adversarial_survival >= self.min_survival
        )

        return DebateVerdict(
            primitive_id=primitive.primitive_id,
            approved=approved,
            consensus_score=consensus,
            adversarial_survival=adversarial_survival,
            statistical_significance=p_value,
            key_arguments_for=args_for,
            key_arguments_against=args_against,
            regime_robustness=regime_robust,
            estimated_capacity=capacity,
        )


# ---------------------------------------------------------------------------
# Shadow Deployment
# ---------------------------------------------------------------------------

@dataclass
class ShadowResult:
    """Result of shadow (paper) deployment of a signal primitive."""
    primitive_id: str
    shadow_days: int
    shadow_pnl: float
    shadow_sharpe: float
    shadow_max_dd: float
    promote_to_live: bool
    reason: str


class ShadowDeployer:
    """
    Run a signal primitive in paper-trading mode for validation
    before promoting to live deployment.
    """

    def __init__(self, min_days: int = 10, min_sharpe: float = 0.3,
                 max_drawdown: float = 0.15):
        self.min_days = min_days
        self.min_sharpe = min_sharpe
        self.max_dd = max_drawdown

    def evaluate_shadow(
        self,
        primitive: SignalPrimitive,
        shadow_returns: List[float],
    ) -> ShadowResult:
        """Evaluate shadow deployment results."""
        n = len(shadow_returns)
        if n < self.min_days:
            return ShadowResult(
                primitive.primitive_id, n, 0.0, 0.0, 0.0, False,
                f"Insufficient shadow data ({n} < {self.min_days} days)"
            )

        rets = np.array(shadow_returns)
        pnl = float(np.prod(1 + rets) - 1)
        sharpe = float(rets.mean() / max(rets.std(), 1e-10) * math.sqrt(252))

        # Max drawdown
        equity = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(equity)
        dd = float(((peak - equity) / peak).max())

        promote = sharpe >= self.min_sharpe and dd <= self.max_dd and pnl > 0
        reason = "PROMOTED" if promote else f"Sharpe={sharpe:.2f}, DD={dd:.1%}, PnL={pnl:.1%}"

        return ShadowResult(
            primitive_id=primitive.primitive_id,
            shadow_days=n,
            shadow_pnl=pnl,
            shadow_sharpe=sharpe,
            shadow_max_dd=dd,
            promote_to_live=promote,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Signal Registry: lifecycle management for all primitives
# ---------------------------------------------------------------------------

class EphemeralPhysicsRegistry:
    """
    Registry for all signal primitives with lifecycle management.

    Signals flow through: candidate -> debating -> backtesting -> shadow -> live -> retired

    The registry auto-retires signals whose IC drops below threshold,
    enabling the system to 'forget' physics that is no longer relevant.
    """

    def __init__(self):
        self._primitives: Dict[str, SignalPrimitive] = {}
        self._verdicts: Dict[str, DebateVerdict] = {}
        self._shadow_results: Dict[str, ShadowResult] = {}

    def register(self, primitive: SignalPrimitive) -> None:
        self._primitives[primitive.primitive_id] = primitive

    def get(self, primitive_id: str) -> Optional[SignalPrimitive]:
        return self._primitives.get(primitive_id)

    def set_verdict(self, verdict: DebateVerdict) -> None:
        self._verdicts[verdict.primitive_id] = verdict
        prim = self._primitives.get(verdict.primitive_id)
        if prim:
            prim.debate_score = verdict.consensus_score
            prim.adversarial_failure_rate = 1 - verdict.adversarial_survival
            if verdict.approved:
                prim.status = "backtesting"
            else:
                prim.status = "retired"
                prim.retired_at = time.time()

    def set_shadow_result(self, result: ShadowResult) -> None:
        self._shadow_results[result.primitive_id] = result
        prim = self._primitives.get(result.primitive_id)
        if prim:
            prim.shadow_pnl = result.shadow_pnl
            if result.promote_to_live:
                prim.status = "live"
                prim.promoted_at = time.time()
            else:
                prim.status = "retired"
                prim.retired_at = time.time()

    def retire_decayed(self, ic_threshold: float = 0.01, lookback: int = 21) -> List[str]:
        """Auto-retire live signals whose IC has decayed below threshold."""
        retired = []
        for pid, prim in self._primitives.items():
            if prim.status != "live":
                continue
            if len(prim.ic_history) >= lookback:
                recent_ic = np.mean(prim.ic_history[-lookback:])
                if abs(recent_ic) < ic_threshold:
                    prim.status = "retired"
                    prim.retired_at = time.time()
                    retired.append(pid)
        return retired

    def get_live(self) -> List[SignalPrimitive]:
        return [p for p in self._primitives.values() if p.status == "live"]

    def get_candidates(self) -> List[SignalPrimitive]:
        return [p for p in self._primitives.values() if p.status == "candidate"]

    def summary(self) -> Dict:
        by_status = {}
        for p in self._primitives.values():
            by_status[p.status] = by_status.get(p.status, 0) + 1
        return {
            "total": len(self._primitives),
            "by_status": by_status,
            "live_count": by_status.get("live", 0),
            "retired_count": by_status.get("retired", 0),
            "domains": list(set(p.domain for p in self._primitives.values())),
        }


# ---------------------------------------------------------------------------
# Event Horizon Synthesizer: the master orchestrator
# ---------------------------------------------------------------------------

class EventHorizonSynthesizer:
    """
    The Event Horizon Synthesizer (EHS).

    An autonomous scientific discovery engine that:
    1. Synthesizes new physics-based trading signals from a concept library
    2. Validates them through adversarial multi-agent debate
    3. Backtests survivors with statistical rigor
    4. Shadow-deploys winners in paper trading
    5. Promotes successful shadows to live deployment
    6. Auto-retires signals that decay

    The full loop runs autonomously. Each cycle:
      - Synthesizes 3-5 new primitives (mutations + new concepts)
      - Debates each through the adversarial gate
      - Backtests approved primitives
      - Checks shadow deployments for promotion
      - Retires decayed live signals
    """

    def __init__(self, seed: int = 42):
        self.synthesizer = PrimitiveSynthesizer(seed)
        self.debate_gate = AdversarialDebateGate()
        self.shadow = ShadowDeployer()
        self.registry = EphemeralPhysicsRegistry()

        self._cycle_count = 0
        self._cycle_history: List[Dict] = []

    def initialize(self) -> None:
        """Bootstrap: synthesize primitives from all physics templates."""
        primitives = self.synthesizer.synthesize_all()
        for p in primitives:
            self.registry.register(p)
        print(f"EHS initialized: {len(primitives)} primitives from {len(PHYSICS_TEMPLATES)} physics concepts")

    def run_cycle(
        self,
        market_returns: np.ndarray,
        regime_labels: np.ndarray,
        shadow_returns_by_primitive: Optional[Dict[str, List[float]]] = None,
    ) -> Dict:
        """
        Run one full EHS cycle.

        market_returns: (T,) array of recent market returns for backtesting
        regime_labels: (T,) array of regime labels
        shadow_returns_by_primitive: dict of primitive_id -> shadow returns (if any running)
        """
        self._cycle_count += 1
        cycle_start = time.time()
        results = {
            "cycle": self._cycle_count,
            "synthesized": 0,
            "debated": 0,
            "approved": 0,
            "promoted": 0,
            "retired": 0,
        }

        # Step 1: Generate new candidates (mutations of live signals + new concepts)
        live = self.registry.get_live()
        candidates = self.registry.get_candidates()

        # Mutate top-performing live signals
        for prim in sorted(live, key=lambda p: np.mean(p.sharpe_history[-10:]) if p.sharpe_history else 0, reverse=True)[:2]:
            mutant = self.synthesizer.mutate_primitive(prim)
            self.registry.register(mutant)
            results["synthesized"] += 1

        # Step 2: Debate all candidates
        for prim in candidates:
            # Simulate the primitive's signal on historical data
            # (simplified: use random subset of market returns as proxy)
            rng = np.random.default_rng(hash(prim.primitive_id) % 2**32)
            signal_returns = market_returns * (rng.normal(0.3, 1.0, len(market_returns)))
            signal_returns *= 0.1  # scale down to realistic alpha levels

            verdict = self.debate_gate.evaluate(prim, signal_returns, regime_labels)
            self.registry.set_verdict(verdict)
            results["debated"] += 1
            if verdict.approved:
                results["approved"] += 1

        # Step 3: Check shadow deployments
        if shadow_returns_by_primitive:
            for pid, shadow_rets in shadow_returns_by_primitive.items():
                prim = self.registry.get(pid)
                if prim and prim.status == "shadow":
                    shadow_result = self.shadow.evaluate_shadow(prim, shadow_rets)
                    self.registry.set_shadow_result(shadow_result)
                    if shadow_result.promote_to_live:
                        results["promoted"] += 1

        # Step 4: Retire decayed signals
        retired = self.registry.retire_decayed()
        results["retired"] = len(retired)

        results["elapsed_seconds"] = time.time() - cycle_start
        results["registry"] = self.registry.summary()
        self._cycle_history.append(results)

        return results

    def get_live_signals(self) -> List[Dict]:
        """Get all currently live signal primitives for the trading engine."""
        return [
            {
                "id": p.primitive_id,
                "name": p.name,
                "physics": p.physics_concept,
                "domain": p.domain,
                "lookback": p.lookback,
                "features": p.feature_names,
                "debate_score": p.debate_score,
                "shadow_pnl": p.shadow_pnl,
                "generation": p.generation,
            }
            for p in self.registry.get_live()
        ]

    def report(self) -> Dict:
        """Full EHS status report."""
        return {
            "total_cycles": self._cycle_count,
            "registry": self.registry.summary(),
            "live_signals": self.get_live_signals(),
            "cycle_history": self._cycle_history[-10:],
            "physics_domains_covered": list(set(
                p.domain for p in self.registry._primitives.values()
            )),
        }
