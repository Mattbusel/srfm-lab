"""
Physics metaphor engine — generates novel hypotheses by drawing analogies
from physics phenomena to market behaviors.

Uses physical laws and phenomena as hypothesis generation templates:
  - Quantum tunneling → price breaking "impossible" resistance
  - Phase transitions → sudden regime changes (vol → price)
  - Resonance → synchronization of correlated assets
  - Damping → momentum decay and mean-reversion
  - Interference → conflicting signals canceling out
  - Entropy → market efficiency cycles
  - Gravity → support/resistance as gravitational fields
  - Conservation laws → capital flow between sectors
"""

from __future__ import annotations
import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

from ..hypothesis.types import Hypothesis, MinedPattern, HypothesisType, HypothesisStatus


@dataclass
class PhysicsMetaphorEngine:
    """
    Generates novel hypotheses by mapping physical phenomena to market dynamics.
    Serendipitous: may find non-obvious patterns via cross-domain analogy.
    """

    seed: int = 42
    n_generate: int = 5

    METAPHORS = [
        "quantum_tunneling",
        "phase_transition",
        "resonance",
        "damping",
        "interference",
        "entropy_maximization",
        "gravitational_lensing",
        "conservation_of_momentum",
        "harmonic_oscillator",
        "critical_phenomena",
        "soliton_propagation",
        "brownian_ratchet",
    ]

    def generate(
        self,
        context: dict[str, Any],
        n: Optional[int] = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses via physics metaphors."""
        rng = random.Random(self.seed + hash(str(context)))
        n = n or self.n_generate

        selected = rng.sample(self.METAPHORS, min(n, len(self.METAPHORS)))
        hypotheses = []

        for metaphor in selected:
            h = self._generate_from_metaphor(metaphor, context, rng)
            if h is not None:
                hypotheses.append(h)

        return hypotheses

    def _generate_from_metaphor(
        self,
        metaphor: str,
        context: dict,
        rng: random.Random,
    ) -> Optional[Hypothesis]:
        symbol = context.get("symbol", "BTC")
        timeframe = context.get("timeframe", "1h")

        if metaphor == "quantum_tunneling":
            return self._quantum_tunneling(symbol, timeframe, context)
        elif metaphor == "phase_transition":
            return self._phase_transition(symbol, timeframe, context)
        elif metaphor == "resonance":
            return self._resonance(symbol, timeframe, context)
        elif metaphor == "damping":
            return self._damping(symbol, timeframe, context)
        elif metaphor == "interference":
            return self._interference(symbol, timeframe, context)
        elif metaphor == "entropy_maximization":
            return self._entropy_max(symbol, timeframe, context)
        elif metaphor == "gravitational_lensing":
            return self._grav_lensing(symbol, timeframe, context)
        elif metaphor == "conservation_of_momentum":
            return self._conservation_momentum(symbol, context)
        elif metaphor == "harmonic_oscillator":
            return self._harmonic_oscillator(symbol, timeframe)
        elif metaphor == "critical_phenomena":
            return self._critical_phenomena(symbol, timeframe, context)
        elif metaphor == "soliton_propagation":
            return self._soliton(symbol, timeframe)
        elif metaphor == "brownian_ratchet":
            return self._brownian_ratchet(symbol, timeframe)
        return None

    def _quantum_tunneling(self, symbol: str, tf: str, ctx: dict) -> Hypothesis:
        resistance = ctx.get("resistance_level", 0.0)
        return Hypothesis(
            id=f"quantum_tunnel_{symbol}",
            name=f"Quantum Tunneling: {symbol} Resistance Break",
            description=(
                f"Physics analogy: quantum tunneling occurs when a particle crosses a barrier "
                f"despite insufficient classical energy — by borrowing from vacuum fluctuations. "
                f"Market analog: {symbol} may break resistance at {resistance:.2f} on low volume "
                f"('quantum' borrowing of liquidity), followed by large move as supply fills gap. "
                f"Hypothesis: after failed resistance touch (price retreats <1%), next attempt "
                f"within 8 bars has 2x higher breakthrough probability."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "resistance_lookback": 20,
                "tunnel_attempt_max_bars": 8,
                "min_failed_touch_retreat": 0.005,
                "max_failed_touch_retreat": 0.02,
                "breakthrough_entry_confirm_bars": 1,
                "stop_below_resistance": 0.02,
            },
            expected_impact=0.03,
            confidence=0.52,
            source_pattern=None,
            tags=["physics_metaphor", "quantum", "breakout", "resistance"],
        )

    def _phase_transition(self, symbol: str, tf: str, ctx: dict) -> Hypothesis:
        order_param = ctx.get("order_parameter", 0.0)
        return Hypothesis(
            id=f"phase_transition_{symbol}",
            name=f"Phase Transition: {symbol} Critical Point Entry",
            description=(
                f"Second-order phase transitions are characterized by critical slowing down "
                f"(variance spikes, autocorrelation → 1) BEFORE the transition. "
                f"When {symbol} shows these early warning signals (rolling AC₁ > 0.7, "
                f"variance 1.5x 90-day avg), a trend phase transition is imminent. "
                f"Enter in the direction of emerging trend with tighter stops."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "timeframe": tf,
                "ac1_critical_threshold": 0.7,
                "variance_ratio_threshold": 1.5,
                "variance_lookback": 90,
                "ac_window": 20,
                "entry_direction": "emerging_trend",
                "stop_loss": 0.03,
            },
            expected_impact=0.04,
            confidence=0.55,
            source_pattern=None,
            tags=["physics_metaphor", "phase_transition", "critical_slowing_down"],
        )

    def _resonance(self, symbol: str, tf: str, ctx: dict) -> Hypothesis:
        correlated_assets = ctx.get("correlated_assets", ["ETH", "BNB"])
        return Hypothesis(
            id=f"resonance_{symbol}",
            name=f"Resonance: Synchronized Asset Signal Amplification",
            description=(
                f"In physics, resonance amplifies when driving frequency matches natural frequency. "
                f"Market analog: when {symbol} AND {correlated_assets} ALL show BH activation "
                f"simultaneously (within 4 bars), the correlated momentum is resonant — "
                f"stronger and more sustained. Size up 1.5x in resonant conditions."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "resonance_assets": correlated_assets[:3],
                "resonance_window_bars": 4,
                "min_resonant_count": 3,
                "size_multiplier_resonant": 1.5,
                "resonance_decay_bars": 12,
            },
            expected_impact=0.025,
            confidence=0.58,
            source_pattern=None,
            tags=["physics_metaphor", "resonance", "cross_asset", "amplification"],
        )

    def _damping(self, symbol: str, tf: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"damping_{symbol}",
            name=f"Overdamping Exit: Momentum Friction Signals Reversal",
            description=(
                f"Overdamped oscillators return to equilibrium without oscillating. "
                f"When {symbol} momentum decays faster than previous trend (momentum half-life "
                f"<50% of entry-bar momentum), system is overdamped — "
                f"price will return to mean without recovering. Exit immediately."
            ),
            hypothesis_type=HypothesisType.EXIT_RULE,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "timeframe": tf,
                "momentum_halflife_threshold": 0.5,
                "momentum_window": 5,
                "momentum_decay_lookback": 10,
                "exit_fraction_overdamped": 0.7,
            },
            expected_impact=0.02,
            confidence=0.54,
            source_pattern=None,
            tags=["physics_metaphor", "damping", "momentum_decay", "exit"],
        )

    def _interference(self, symbol: str, tf: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"interference_{symbol}",
            name=f"Destructive Interference: Conflicting Signal Avoidance",
            description=(
                f"Destructive interference cancels wave amplitude. "
                f"When {symbol} BH signal is bullish but RSI is overbought AND funding is extreme: "
                f"signals are in destructive interference — net force near zero. "
                f"Skip entries when ≥2 independent signals directly conflict."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "min_conflicting_signals": 2,
                "bh_rsi_conflict_threshold": 75,
                "bh_funding_conflict_threshold": 0.001,
                "skip_entry_on_interference": True,
            },
            expected_impact=0.015,
            confidence=0.60,
            source_pattern=None,
            tags=["physics_metaphor", "interference", "signal_conflict", "filter"],
        )

    def _entropy_max(self, symbol: str, tf: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"entropy_max_{symbol}",
            name=f"Maximum Entropy Sizing: Uncertainty-Scaled Position",
            description=(
                f"Maximum entropy principle: given partial information, "
                f"the least biased distribution maximizes entropy. "
                f"Apply to position sizing: when signal confidence is moderate (40-60%), "
                f"use entropy-maximizing weights rather than binary in/out. "
                f"Size = confidence * max_size * (1 - H_normalized) where H = signal entropy."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "entropy_based_sizing": True,
                "entropy_window": 30,
                "max_size_multiplier": 1.0,
                "min_size_multiplier": 0.2,
            },
            expected_impact=0.01,
            confidence=0.55,
            source_pattern=None,
            tags=["physics_metaphor", "entropy", "position_sizing", "maximum_entropy"],
        )

    def _grav_lensing(self, symbol: str, tf: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"grav_lensing_{symbol}",
            name=f"Gravitational Lensing: BH Bends Signal Paths",
            description=(
                f"Gravitational lensing bends light paths near massive objects. "
                f"In our BH physics framework: a large BH mass in {symbol} 'bends' "
                f"correlated assets' price paths toward it. "
                f"When BH mass > 1.9 in {symbol}, fade short-term reversals in correlated alts "
                f"— they will be pulled back toward {symbol}'s trajectory."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "bh_mass_lensing_threshold": 1.9,
                "correlated_assets_max": 3,
                "lensing_entry_retracement": 0.02,
                "lensing_hold_bars": 8,
            },
            expected_impact=0.02,
            confidence=0.53,
            source_pattern=None,
            tags=["physics_metaphor", "gravitational_lensing", "black_hole", "cross_asset"],
        )

    def _conservation_momentum(self, symbol: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"conservation_momentum_{symbol}",
            name="Conservation of Momentum: Sector Rotation Tracker",
            description=(
                "Physics: total momentum in a closed system is conserved. "
                "Market analog: total crypto market cap momentum is roughly conserved short-term. "
                "When BTC dumps, alt momentum is created. "
                "Track net momentum transfer from BTC to alts as entry signal for altcoin bounce."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "btc_dump_threshold": -0.05,
                "btc_dump_window_bars": 4,
                "alt_momentum_lag": 2,
                "target_symbols": ["ETH", "BNB", "SOL", "XRP"],
                "min_alt_underperformance": 0.03,
            },
            expected_impact=0.025,
            confidence=0.56,
            source_pattern=None,
            tags=["physics_metaphor", "conservation", "sector_rotation", "cross_asset"],
        )

    def _harmonic_oscillator(self, symbol: str, tf: str) -> Hypothesis:
        return Hypothesis(
            id=f"harmonic_osc_{symbol}",
            name=f"Harmonic Oscillator: Natural Frequency Entry",
            description=(
                f"Damped harmonic oscillators have a natural frequency of oscillation. "
                f"Estimate {symbol}'s dominant cycle period via wavelet analysis. "
                f"Enter at oscillator troughs (price at -1σ of cycle), "
                f"exit at crests (+1σ). Use cycle period as hold time target."
            ),
            hypothesis_type=HypothesisType.ENTRY_TIMING,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "timeframe": tf,
                "wavelet_scales": [8, 16, 32, 64],
                "dominant_cycle_detection": True,
                "trough_entry_threshold": -1.0,
                "crest_exit_threshold": 1.0,
                "cycle_period_as_hold": True,
            },
            expected_impact=0.02,
            confidence=0.54,
            source_pattern=None,
            tags=["physics_metaphor", "oscillator", "cycle", "wavelet"],
        )

    def _critical_phenomena(self, symbol: str, tf: str, ctx: dict) -> Hypothesis:
        return Hypothesis(
            id=f"critical_phenomena_{symbol}",
            name=f"Critical Phenomena: Power-Law Precursor Detection",
            description=(
                f"Near critical points, correlation functions follow power laws with "
                f"universal critical exponents. "
                f"In markets: log-periodic oscillations in price precede crashes (LPPLS model). "
                f"Detect LPPLS fit in {symbol} — if fit quality > 0.8, reduce position by 50%."
            ),
            hypothesis_type=HypothesisType.REGIME_FILTER,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "timeframe": tf,
                "lppls_fit_threshold": 0.8,
                "lppls_lookback_bars": 120,
                "position_reduction_on_fit": 0.5,
                "critical_time_warning_bars": 20,
            },
            expected_impact=0.03,
            confidence=0.57,
            source_pattern=None,
            tags=["physics_metaphor", "critical_phenomena", "power_law", "crash_detection"],
        )

    def _soliton(self, symbol: str, tf: str) -> Hypothesis:
        return Hypothesis(
            id=f"soliton_{symbol}",
            name=f"Soliton Wave: Self-Reinforcing Price Wave",
            description=(
                f"Solitons are stable, self-reinforcing waves that maintain shape over distance. "
                f"Market analog: when {symbol} BH mass grows monotonically for 20+ bars "
                f"WITHOUT a mass contraction >5%, the BH is a soliton-like structure — "
                f"extremely stable, rides through noise. Increase position to full size in soliton."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "timeframe": tf,
                "soliton_min_bars": 20,
                "max_contraction_pct": 0.05,
                "soliton_size_multiplier": 1.5,
                "soliton_exit_on_first_contraction": False,
            },
            expected_impact=0.03,
            confidence=0.58,
            source_pattern=None,
            tags=["physics_metaphor", "soliton", "black_hole", "trend_strength"],
        )

    def _brownian_ratchet(self, symbol: str, tf: str) -> Hypothesis:
        return Hypothesis(
            id=f"brownian_ratchet_{symbol}",
            name=f"Brownian Ratchet: Asymmetric Vol Extraction",
            description=(
                f"A Brownian ratchet extracts work from thermal noise via asymmetric potential. "
                f"Market analog: use asymmetric rebalancing — buy dips faster than selling rallies. "
                f"When {symbol} drops 1.5% intrabar, add 20% to position; "
                f"when up 2%, sell 10%. Net: ratchet momentum trades in profitable direction."
            ),
            hypothesis_type=HypothesisType.PARAMETER_TWEAK,
            status=HypothesisStatus.PENDING,
            parameters={
                "symbol": symbol,
                "timeframe": tf,
                "dip_add_threshold": -0.015,
                "dip_add_fraction": 0.2,
                "rally_trim_threshold": 0.02,
                "rally_trim_fraction": 0.1,
                "max_ratchet_adds": 3,
            },
            expected_impact=0.015,
            confidence=0.55,
            source_pattern=None,
            tags=["physics_metaphor", "ratchet", "asymmetric", "rebalancing"],
        )
