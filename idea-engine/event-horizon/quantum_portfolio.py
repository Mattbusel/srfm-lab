"""
Quantum Portfolio: superposition-based position management.

Instead of discrete positions (long OR short), maintain portfolios
in SUPERPOSITION: simultaneously consider multiple possible positions
and only "collapse" to a discrete state when you must execute.

Inspired by quantum computing's ability to explore exponentially
many states simultaneously:

1. Each position exists as a probability amplitude across states
   (long, short, flat) with complex-valued weights
2. Portfolio "entanglement": correlated positions collapse together
3. Measurement = execution: only when you place an order does the
   quantum state collapse to a classical position
4. Uncertainty principle: the more precisely you know direction,
   the less precisely you know timing (and vice versa)

Practical application: instead of binary "long/short" decisions,
maintain a DISTRIBUTION of possible positions and only commit
when the distribution is sufficiently concentrated.
"""

from __future__ import annotations
import math
import cmath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QuantumState:
    """
    A position in superposition.

    |psi> = alpha_long|long> + alpha_flat|flat> + alpha_short|short>

    where alpha_i are complex amplitudes and |alpha_i|^2 are probabilities.
    """
    symbol: str
    alpha_long: complex = 0.577 + 0j    # equal superposition default
    alpha_flat: complex = 0.577 + 0j
    alpha_short: complex = 0.577 + 0j

    @property
    def p_long(self) -> float:
        return float(abs(self.alpha_long) ** 2)

    @property
    def p_flat(self) -> float:
        return float(abs(self.alpha_flat) ** 2)

    @property
    def p_short(self) -> float:
        return float(abs(self.alpha_short) ** 2)

    @property
    def expected_position(self) -> float:
        """Expected value: +1*p_long + 0*p_flat + (-1)*p_short."""
        return self.p_long - self.p_short

    @property
    def uncertainty(self) -> float:
        """Variance of the position distribution."""
        exp = self.expected_position
        var = self.p_long * (1 - exp)**2 + self.p_flat * (0 - exp)**2 + self.p_short * (-1 - exp)**2
        return float(math.sqrt(max(var, 0)))

    @property
    def entropy(self) -> float:
        """Shannon entropy of the state (0 = collapsed, max = equal superposition)."""
        probs = [self.p_long, self.p_flat, self.p_short]
        return float(-sum(p * math.log(max(p, 1e-15)) for p in probs if p > 1e-15))

    def normalize(self) -> None:
        """Ensure probabilities sum to 1."""
        norm = math.sqrt(abs(self.alpha_long)**2 + abs(self.alpha_flat)**2 + abs(self.alpha_short)**2)
        if norm > 1e-10:
            self.alpha_long /= norm
            self.alpha_flat /= norm
            self.alpha_short /= norm

    def is_collapsed(self, threshold: float = 0.8) -> bool:
        """Has the state effectively collapsed to a single position?"""
        return max(self.p_long, self.p_flat, self.p_short) > threshold

    def collapsed_state(self) -> str:
        """Which state has the highest probability?"""
        probs = {"long": self.p_long, "flat": self.p_flat, "short": self.p_short}
        return max(probs, key=probs.get)


class QuantumGate:
    """
    Operations that evolve quantum portfolio states.

    Each signal is a "quantum gate" that rotates the state vector:
    - Bullish signal: rotates toward |long>
    - Bearish signal: rotates toward |short>
    - Uncertain signal: rotates toward |flat> (superposition)
    """

    @staticmethod
    def signal_gate(state: QuantumState, signal: float, strength: float = 0.1) -> QuantumState:
        """
        Apply a signal as a rotation gate.
        signal > 0: rotate toward long
        signal < 0: rotate toward short
        strength: how much to rotate (0=no change, 1=full collapse)
        """
        theta = signal * strength * math.pi / 2

        if signal > 0:
            # Rotate from flat/short toward long
            state.alpha_long = state.alpha_long * cmath.exp(1j * theta) + complex(abs(signal) * strength * 0.3, 0)
            state.alpha_short = state.alpha_short * cmath.exp(-1j * theta * 0.5)
        else:
            # Rotate from flat/long toward short
            state.alpha_short = state.alpha_short * cmath.exp(1j * theta) + complex(abs(signal) * strength * 0.3, 0)
            state.alpha_long = state.alpha_long * cmath.exp(-1j * theta * 0.5)

        state.normalize()
        return state

    @staticmethod
    def decoherence(state: QuantumState, rate: float = 0.01) -> QuantumState:
        """
        Natural decoherence: over time, quantum states decay toward
        equal superposition (maximum uncertainty).

        This models the "forgetting" of old signal information.
        """
        target = complex(1 / math.sqrt(3), 0)
        state.alpha_long = state.alpha_long * (1 - rate) + target * rate
        state.alpha_flat = state.alpha_flat * (1 - rate) + target * rate
        state.alpha_short = state.alpha_short * (1 - rate) + target * rate
        state.normalize()
        return state

    @staticmethod
    def entangle(state_a: QuantumState, state_b: QuantumState,
                  correlation: float) -> Tuple[QuantumState, QuantumState]:
        """
        Entangle two positions: if they are correlated, collapsing one
        should influence the other.

        High positive correlation: states align
        High negative correlation: states anti-align
        """
        # Mix amplitudes based on correlation
        mix = abs(correlation) * 0.3

        if correlation > 0:
            # Positive correlation: align states
            state_b.alpha_long = state_b.alpha_long * (1 - mix) + state_a.alpha_long * mix
            state_b.alpha_short = state_b.alpha_short * (1 - mix) + state_a.alpha_short * mix
        else:
            # Negative correlation: anti-align
            state_b.alpha_long = state_b.alpha_long * (1 - mix) + state_a.alpha_short * mix
            state_b.alpha_short = state_b.alpha_short * (1 - mix) + state_a.alpha_long * mix

        state_a.normalize()
        state_b.normalize()
        return state_a, state_b


class QuantumPortfolioManager:
    """
    Manage a portfolio of quantum positions.

    Each position is in superposition until execution.
    Only when confidence (state collapse) exceeds threshold do we trade.

    Benefits:
    - Avoids premature commitment to a direction
    - Naturally represents uncertainty in position sizing
    - Entanglement captures cross-asset dependencies
    - Decoherence prevents stale signals from persisting
    """

    def __init__(self, symbols: List[str], collapse_threshold: float = 0.7,
                  decoherence_rate: float = 0.02):
        self.symbols = symbols
        self.threshold = collapse_threshold
        self.decoherence_rate = decoherence_rate

        self.states: Dict[str, QuantumState] = {
            sym: QuantumState(symbol=sym) for sym in symbols
        }
        self._collapsed_positions: Dict[str, str] = {}  # symbol -> "long"/"short"/"flat"

    def apply_signal(self, symbol: str, signal: float, strength: float = 0.15) -> None:
        """Apply a trading signal as a quantum gate rotation."""
        if symbol in self.states:
            QuantumGate.signal_gate(self.states[symbol], signal, strength)

    def apply_entanglement(self, correlations: Dict[Tuple[str, str], float]) -> None:
        """Apply entanglement between correlated positions."""
        for (sym_a, sym_b), corr in correlations.items():
            if sym_a in self.states and sym_b in self.states:
                QuantumGate.entangle(self.states[sym_a], self.states[sym_b], corr)

    def evolve(self) -> None:
        """One time step: apply decoherence to all positions."""
        for state in self.states.values():
            QuantumGate.decoherence(state, self.decoherence_rate)

    def measure(self) -> Dict[str, Dict]:
        """
        "Measure" the portfolio: check which positions have collapsed
        and return the current quantum state of each.
        """
        results = {}
        for sym, state in self.states.items():
            collapsed = state.is_collapsed(self.threshold)

            results[sym] = {
                "p_long": state.p_long,
                "p_flat": state.p_flat,
                "p_short": state.p_short,
                "expected_position": state.expected_position,
                "uncertainty": state.uncertainty,
                "entropy": state.entropy,
                "collapsed": collapsed,
                "collapsed_to": state.collapsed_state() if collapsed else "superposition",
            }

            if collapsed and sym not in self._collapsed_positions:
                self._collapsed_positions[sym] = state.collapsed_state()

        return results

    def get_executable_trades(self) -> List[Dict]:
        """
        Get only the positions that have collapsed enough to trade.
        These are high-confidence, low-uncertainty positions.
        """
        trades = []
        for sym, state in self.states.items():
            if state.is_collapsed(self.threshold):
                direction = state.collapsed_state()
                if direction != "flat":
                    confidence = max(state.p_long, state.p_short)
                    trades.append({
                        "symbol": sym,
                        "direction": direction,
                        "confidence": confidence,
                        "uncertainty": state.uncertainty,
                        "size_fraction": confidence * 0.1,  # scale by confidence
                    })
        return trades

    def get_portfolio_entropy(self) -> float:
        """Total portfolio uncertainty."""
        return float(np.mean([s.entropy for s in self.states.values()]))

    def get_status(self) -> Dict:
        n_collapsed = sum(1 for s in self.states.values() if s.is_collapsed(self.threshold))
        avg_uncertainty = float(np.mean([s.uncertainty for s in self.states.values()]))

        return {
            "n_positions": len(self.states),
            "n_collapsed": n_collapsed,
            "n_superposition": len(self.states) - n_collapsed,
            "avg_uncertainty": avg_uncertainty,
            "portfolio_entropy": self.get_portfolio_entropy(),
            "executable_trades": len(self.get_executable_trades()),
        }
