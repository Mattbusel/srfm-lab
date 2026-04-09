"""
Market Consciousness Model: emergent beliefs from multi-agent neural consensus.

Treats the debate system as a recurrent neural network:
  - Each agent is a neuron with activation = conviction
  - Cross-domain mappings define synaptic weights
  - Debate history forms recurrent memory (hidden state)
  - Consensus is the network's collective activation
  - Emergent beliefs = states that no individual agent holds but the
    collective network converges on

The key insight: market regime transitions are preceded by measurable
increases in mutual information between disparate agent outputs.
When the microstructure agent and the thermodynamics agent suddenly
agree despite operating on completely different data, something
is about to happen.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Neural Agent: wraps a debate agent as a neuron
# ---------------------------------------------------------------------------

@dataclass
class NeuralAgent:
    """A debate agent modeled as a neuron in the consciousness network."""
    name: str
    domain: str               # physics / microstructure / macro / ml / risk / regime
    activation: float = 0.0   # current conviction output (-1 to +1)
    bias: float = 0.0         # persistent bias from historical accuracy
    firing_history: deque = field(default_factory=lambda: deque(maxlen=100))
    accuracy_ema: float = 0.5

    def fire(self, conviction: float) -> float:
        """Neuron fires: applies activation function to conviction."""
        self.activation = math.tanh(conviction + self.bias)
        self.firing_history.append(self.activation)
        return self.activation

    def update_accuracy(self, was_correct: bool, lr: float = 0.05) -> None:
        """Update bias based on prediction accuracy (Hebbian-like learning)."""
        target = 1.0 if was_correct else 0.0
        self.accuracy_ema = (1 - lr) * self.accuracy_ema + lr * target
        # Bias adjustment: accurate agents get amplified
        self.bias = (self.accuracy_ema - 0.5) * 0.3


# ---------------------------------------------------------------------------
# Synaptic Connections: cross-domain coupling
# ---------------------------------------------------------------------------

# Which agent domains should have strong synaptic connections?
# These represent conceptual bridges between domains.
SYNAPTIC_WEIGHTS = {
    ("physics", "microstructure"): 0.7,     # BH mass links to order flow
    ("physics", "regime"): 0.8,             # physics drives regime detection
    ("microstructure", "risk"): 0.6,        # order flow signals risk
    ("macro", "regime"): 0.7,               # macro drives regime
    ("macro", "risk"): 0.5,                 # macro signals systemic risk
    ("ml", "physics"): 0.4,                 # ML can detect physics patterns
    ("ml", "microstructure"): 0.5,          # ML finds microstructure patterns
    ("regime", "risk"): 0.8,               # regime directly impacts risk
    ("physics", "macro"): 0.3,             # physics-macro cross-pollination
    ("microstructure", "regime"): 0.6,     # microstructure reveals regime
}


def get_synaptic_weight(domain_a: str, domain_b: str) -> float:
    """Get the synaptic weight between two agent domains."""
    if domain_a == domain_b:
        return 1.0  # self-connection
    key = tuple(sorted([domain_a, domain_b]))
    return SYNAPTIC_WEIGHTS.get(key, 0.1)  # weak default connection


# ---------------------------------------------------------------------------
# Debate Trace Buffer: recurrent memory
# ---------------------------------------------------------------------------

@dataclass
class ConsensusState:
    """A snapshot of the network's state at one debate round."""
    timestamp: float
    activations: Dict[str, float]     # agent_name -> activation
    collective_activation: float       # network output
    agreement_level: float            # how much agents agree (0=split, 1=unanimous)
    entropy: float                    # Shannon entropy of activation distribution
    dominant_domain: str              # which domain is driving consensus
    emerging_belief: str              # human-readable emergent belief


class DebateTraceBuffer:
    """
    Recurrent memory buffer storing the hidden state of the consciousness network.
    Enables detection of emergent patterns that no single debate round reveals.
    """

    def __init__(self, max_length: int = 200):
        self._states: deque = deque(maxlen=max_length)
        self._belief_transitions: List[Tuple[str, str, float]] = []

    def append(self, state: ConsensusState) -> None:
        # Detect belief transitions
        if self._states:
            prev = self._states[-1]
            if prev.emerging_belief != state.emerging_belief:
                self._belief_transitions.append(
                    (prev.emerging_belief, state.emerging_belief, state.timestamp)
                )
        self._states.append(state)

    def get_hidden_state(self, window: int = 10) -> np.ndarray:
        """
        Extract the 'hidden state' of the RNN: a vector summarizing
        recent debate history.
        """
        if len(self._states) < 3:
            return np.zeros(6)

        recent = list(self._states)[-window:]

        # Hidden state components
        avg_activation = float(np.mean([s.collective_activation for s in recent]))
        activation_trend = float(np.polyfit(range(len(recent)),
                                            [s.collective_activation for s in recent], 1)[0])
        avg_agreement = float(np.mean([s.agreement_level for s in recent]))
        avg_entropy = float(np.mean([s.entropy for s in recent]))
        n_transitions = len([t for t in self._belief_transitions if t[2] > recent[0].timestamp])
        stability = 1.0 / max(n_transitions + 1, 1)

        return np.array([avg_activation, activation_trend, avg_agreement,
                          avg_entropy, n_transitions, stability])

    def detect_phase_transition(self, window: int = 20) -> Optional[Dict]:
        """
        Detect sudden changes in the consensus dynamics that signal
        an impending market regime shift.

        A "phase transition" in the consciousness model occurs when:
        1. Agreement suddenly spikes (agents converge from disagreement)
        2. Entropy drops sharply (uncertainty collapses)
        3. Dominant domain changes (new information source takes over)
        """
        if len(self._states) < window + 5:
            return None

        recent = list(self._states)[-window:]
        early = recent[:window // 2]
        late = recent[window // 2:]

        # Agreement spike
        early_agreement = np.mean([s.agreement_level for s in early])
        late_agreement = np.mean([s.agreement_level for s in late])
        agreement_spike = late_agreement - early_agreement

        # Entropy drop
        early_entropy = np.mean([s.entropy for s in early])
        late_entropy = np.mean([s.entropy for s in late])
        entropy_drop = early_entropy - late_entropy

        # Domain shift
        early_domains = [s.dominant_domain for s in early]
        late_domains = [s.dominant_domain for s in late]
        domain_changed = early_domains[-1] != late_domains[-1]

        # Phase transition score
        score = agreement_spike * 2 + entropy_drop * 3 + (1.0 if domain_changed else 0.0)

        if score > 1.5:
            return {
                "detected": True,
                "score": float(score),
                "agreement_spike": float(agreement_spike),
                "entropy_drop": float(entropy_drop),
                "domain_shift": domain_changed,
                "from_domain": early_domains[-1],
                "to_domain": late_domains[-1],
                "from_belief": early[-1].emerging_belief,
                "to_belief": late[-1].emerging_belief,
                "implication": "Market regime transition imminent" if score > 2.5
                               else "Increased conviction in current view",
            }
        return None


# ---------------------------------------------------------------------------
# Mutual Information Monitor
# ---------------------------------------------------------------------------

class MutualInformationMonitor:
    """
    Track mutual information between agent pairs.

    When two agents that normally operate independently suddenly show
    high MI, it indicates they're detecting the same underlying phenomenon
    from different perspectives -- a strong conviction signal.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._agent_histories: Dict[str, deque] = {}

    def record(self, agent_name: str, activation: float) -> None:
        if agent_name not in self._agent_histories:
            self._agent_histories[agent_name] = deque(maxlen=100)
        self._agent_histories[agent_name].append(activation)

    def compute_mi(self, agent_a: str, agent_b: str) -> float:
        """Compute mutual information between two agents' activation histories."""
        hist_a = list(self._agent_histories.get(agent_a, []))
        hist_b = list(self._agent_histories.get(agent_b, []))
        n = min(len(hist_a), len(hist_b))
        if n < 20:
            return 0.0

        a = np.array(hist_a[-n:])
        b = np.array(hist_b[-n:])

        # Discretize into bins
        a_bins = np.digitize(a, np.linspace(a.min() - 1e-10, a.max() + 1e-10, self.n_bins))
        b_bins = np.digitize(b, np.linspace(b.min() - 1e-10, b.max() + 1e-10, self.n_bins))

        # Joint distribution
        joint = np.zeros((self.n_bins + 1, self.n_bins + 1))
        for ai, bi in zip(a_bins, b_bins):
            joint[ai, bi] += 1
        joint /= joint.sum() + 1e-10

        # Marginals
        p_a = joint.sum(axis=1)
        p_b = joint.sum(axis=0)

        # MI = sum p(a,b) * log(p(a,b) / (p(a) * p(b)))
        mi = 0.0
        for i in range(self.n_bins + 1):
            for j in range(self.n_bins + 1):
                if joint[i, j] > 1e-10 and p_a[i] > 1e-10 and p_b[j] > 1e-10:
                    mi += joint[i, j] * math.log(joint[i, j] / (p_a[i] * p_b[j]))

        return max(0.0, mi)

    def find_surprising_agreements(self, threshold: float = 0.3) -> List[Dict]:
        """Find agent pairs with unexpectedly high mutual information."""
        agents = list(self._agent_histories.keys())
        surprises = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                mi = self.compute_mi(agents[i], agents[j])
                if mi > threshold:
                    surprises.append({
                        "agent_a": agents[i],
                        "agent_b": agents[j],
                        "mutual_information": mi,
                        "implication": f"{agents[i]} and {agents[j]} are detecting the same phenomenon",
                    })
        surprises.sort(key=lambda x: x["mutual_information"], reverse=True)
        return surprises


# ---------------------------------------------------------------------------
# Market Consciousness Engine
# ---------------------------------------------------------------------------

class MarketConsciousness:
    """
    The consciousness network: multi-agent RNN with emergent beliefs.

    Usage:
      mc = MarketConsciousness(agent_names_and_domains)
      mc.process_debate(agent_convictions)  # each debate round
      belief = mc.get_emergent_belief()     # what does the network believe?
      transition = mc.detect_phase_transition()  # is a regime shift coming?
    """

    def __init__(self, agents: Dict[str, str]):
        """agents: dict of agent_name -> domain"""
        self.neurons = {name: NeuralAgent(name, domain) for name, domain in agents.items()}
        self.trace = DebateTraceBuffer()
        self.mi_monitor = MutualInformationMonitor()
        self._cycle = 0

    def process_debate(self, convictions: Dict[str, float], timestamp: float = 0.0) -> ConsensusState:
        """
        Process one debate round through the consciousness network.

        convictions: dict of agent_name -> conviction value (-1 to +1)
        """
        self._cycle += 1
        if timestamp == 0.0:
            timestamp = float(self._cycle)

        # Fire each neuron
        activations = {}
        for name, conviction in convictions.items():
            if name in self.neurons:
                act = self.neurons[name].fire(conviction)
                activations[name] = act
                self.mi_monitor.record(name, act)

        # Compute collective activation (synaptic-weighted sum)
        total_weighted = 0.0
        total_weight = 0.0
        for name_a, act_a in activations.items():
            for name_b, act_b in activations.items():
                if name_a >= name_b:
                    continue
                w = get_synaptic_weight(
                    self.neurons[name_a].domain,
                    self.neurons[name_b].domain,
                )
                # Coupled activation: strong when both agree
                coupled = act_a * act_b * w
                total_weighted += coupled
                total_weight += w

        collective = float(math.tanh(total_weighted / max(total_weight, 1e-10) * 3))

        # Agreement level: how much do agents agree?
        acts = list(activations.values())
        if acts:
            signs = [1 if a > 0 else -1 for a in acts if abs(a) > 0.1]
            if signs:
                agreement = abs(sum(signs)) / len(signs)
            else:
                agreement = 0.0
        else:
            agreement = 0.0

        # Entropy of activation distribution
        abs_acts = [abs(a) for a in acts if abs(a) > 0.01]
        if abs_acts:
            total = sum(abs_acts)
            probs = [a / total for a in abs_acts]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
            normalized_entropy = entropy / max(max_entropy, 1e-10)
        else:
            normalized_entropy = 1.0

        # Dominant domain
        domain_activations = {}
        for name, act in activations.items():
            domain = self.neurons[name].domain
            domain_activations[domain] = domain_activations.get(domain, 0) + abs(act)
        dominant = max(domain_activations, key=domain_activations.get) if domain_activations else "none"

        # Emergent belief: a qualitative description of what the network believes
        if collective > 0.5 and agreement > 0.6:
            belief = f"Strong bullish consensus (driven by {dominant})"
        elif collective < -0.5 and agreement > 0.6:
            belief = f"Strong bearish consensus (driven by {dominant})"
        elif agreement < 0.3:
            belief = f"Disagreement (no consensus, entropy={normalized_entropy:.2f})"
        elif normalized_entropy < 0.3:
            belief = f"Emerging conviction ({dominant} leading)"
        else:
            belief = f"Neutral (weak signals from {dominant})"

        state = ConsensusState(
            timestamp=timestamp,
            activations=activations,
            collective_activation=collective,
            agreement_level=float(agreement),
            entropy=float(normalized_entropy),
            dominant_domain=dominant,
            emerging_belief=belief,
        )

        self.trace.append(state)
        return state

    def get_emergent_belief(self) -> Dict:
        """What does the network currently believe?"""
        if not self.trace._states:
            return {"belief": "no data", "confidence": 0.0}

        latest = self.trace._states[-1]
        hidden = self.trace.get_hidden_state()

        return {
            "belief": latest.emerging_belief,
            "collective_activation": latest.collective_activation,
            "agreement": latest.agreement_level,
            "entropy": latest.entropy,
            "dominant_domain": latest.dominant_domain,
            "hidden_state": hidden.tolist(),
            "cycle": self._cycle,
        }

    def detect_phase_transition(self) -> Optional[Dict]:
        """Is a market regime transition imminent?"""
        return self.trace.detect_phase_transition()

    def get_surprising_agreements(self) -> List[Dict]:
        """Which agents that normally disagree are suddenly agreeing?"""
        return self.mi_monitor.find_surprising_agreements()

    def get_signal(self) -> float:
        """
        Convert the consciousness state into a trading signal.
        Range: [-1, +1]
        """
        if not self.trace._states:
            return 0.0

        latest = self.trace._states[-1]
        hidden = self.trace.get_hidden_state()

        # Signal components
        conviction = latest.collective_activation
        agreement_boost = latest.agreement_level ** 2  # quadratic: strong agreement amplifies
        entropy_confidence = 1 - latest.entropy  # low entropy = high confidence

        # Phase transition detection amplifies signal
        transition = self.trace.detect_phase_transition()
        transition_boost = 1.0
        if transition:
            transition_boost = 1.0 + transition["score"] * 0.3

        signal = conviction * agreement_boost * entropy_confidence * transition_boost
        return float(np.clip(signal, -1, 1))
