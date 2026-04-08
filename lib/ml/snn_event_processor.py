"""
Spiking Neural Network Event Processor (T4-2) — SCAFFOLD
Asynchronous event-driven signal processing using Spiking Neural Networks.

Architecture: Liquid State Machine (LSM) with leaky integrate-and-fire neurons.
Processes market events as spikes rather than waiting for bar closes.

Financial events → spikes:
  - Price threshold crossing → isolated spike
  - Volume surge (>2σ) → burst of 3 spikes
  - BH formation → population burst (all neurons spike)

This scaffold implements the SNN architecture. Full production deployment
requires tick data feed (significant infrastructure investment).

Status: RESEARCH SCAFFOLD — not connected to live trading loop yet.
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
import random

log = logging.getLogger(__name__)

@dataclass
class LIFNeuronConfig:
    tau_m: float = 20.0       # membrane time constant (ms equivalent)
    tau_s: float = 5.0        # synaptic time constant
    v_thresh: float = 1.0     # spike threshold
    v_reset: float = 0.0      # post-spike reset potential
    v_rest: float = 0.0       # resting potential
    refractory_period: int = 2  # bars of refractory period after spike

@dataclass
class LSMConfig:
    n_input: int = 6          # input neurons (price_change, volume, mass, beta, cf, hurst)
    n_reservoir: int = 100    # reservoir neurons
    n_readout: int = 4        # readout neurons (bh_prob, exit_prob, size_scale, regime)
    spectral_radius: float = 0.9  # reservoir weight spectral radius (< 1 for stability)
    input_sparsity: float = 0.3   # fraction of inputs connected to each reservoir neuron
    reservoir_sparsity: float = 0.1  # fraction of reservoir-reservoir connections
    seed: int = 42

class LIFNeuron:
    """Leaky Integrate-and-Fire neuron."""

    def __init__(self, cfg: LIFNeuronConfig = None):
        self.cfg = cfg or LIFNeuronConfig()
        self.v: float = self.cfg.v_rest     # membrane potential
        self.spiked: bool = False
        self._refractory: int = 0
        self.spike_history: list[int] = []   # timesteps of spikes

    def update(self, current: float, timestep: int) -> bool:
        """Update neuron state. Returns True if neuron spiked."""
        if self._refractory > 0:
            self._refractory -= 1
            self.spiked = False
            return False

        # Leaky integration: dV/dt = -(V - V_rest)/τ_m + I
        alpha = 1.0 - 1.0 / self.cfg.tau_m
        self.v = alpha * self.v + (1 - alpha) * self.cfg.v_rest + current

        if self.v >= self.cfg.v_thresh:
            self.v = self.cfg.v_reset
            self.spiked = True
            self._refractory = self.cfg.refractory_period
            self.spike_history.append(timestep)
            if len(self.spike_history) > 100:
                self.spike_history = self.spike_history[-100:]
            return True

        self.spiked = False
        return False

    def firing_rate(self, window: int = 20) -> float:
        """Spikes per timestep in last window."""
        if not self.spike_history:
            return 0.0
        recent = sum(1 for t in self.spike_history if self.spike_history[-1] - t < window)
        return recent / window

class LiquidStateMachine:
    """
    Liquid State Machine for market event processing.

    Reservoir provides rich temporal representation of spike trains.
    Readout layer (linear decoder) trained to predict trading signals.

    This is a research scaffold — readout weights are initialized randomly
    and require proper training (FORCE learning or backprop through BPTT).
    """

    def __init__(self, cfg: LSMConfig = None):
        self.cfg = cfg or LSMConfig()
        self._rng = random.Random(self.cfg.seed)
        self._timestep: int = 0

        # Create neurons
        self._reservoir = [LIFNeuron() for _ in range(self.cfg.n_reservoir)]

        # Input → reservoir weights (sparse)
        self._W_in = self._init_sparse_weights(
            self.cfg.n_input, self.cfg.n_reservoir, self.cfg.input_sparsity
        )

        # Reservoir → reservoir weights (sparse, scaled by spectral radius)
        self._W_res = self._init_reservoir_weights()

        # Readout weights (reservoir → output): randomly initialized, needs training
        self._W_out = [
            [self._rng.gauss(0, 0.1) for _ in range(self.cfg.n_reservoir)]
            for _ in range(self.cfg.n_readout)
        ]

        log.info("LSM: initialized %d reservoir neurons, %d inputs, %d readouts",
                 self.cfg.n_reservoir, self.cfg.n_input, self.cfg.n_readout)

    def process_spike_train(
        self,
        price_change: float,
        volume_zscore: float,
        bh_mass: float,
        beta: float,
        cf: float,
        hurst: float,
    ) -> dict:
        """
        Process market state as input spike train.

        Returns readout activations:
          bh_prob: probability of BH formation
          exit_prob: probability of exit signal
          size_scale: position size suggestion
          regime: regime encoding (0=bull, 1=bear, 2=sideways)
        """
        self._timestep += 1

        # Encode inputs as spike rates (rate coding)
        inputs = self._encode_inputs(price_change, volume_zscore, bh_mass, beta, cf, hurst)

        # Compute reservoir state
        reservoir_state = []
        for j, neuron in enumerate(self._reservoir):
            # Sum of input currents
            input_current = sum(self._W_in[i][j] * inputs[i] for i in range(self.cfg.n_input))

            # Sum of recurrent currents (from previous timestep)
            recurrent_current = sum(
                self._W_res[k][j] * self._reservoir[k].spiked
                for k in range(self.cfg.n_reservoir)
            )

            neuron.update(input_current + recurrent_current, self._timestep)
            reservoir_state.append(neuron.firing_rate())

        # Readout: linear combination of reservoir firing rates
        readout = []
        for i in range(self.cfg.n_readout):
            activation = sum(self._W_out[i][j] * reservoir_state[j]
                           for j in range(self.cfg.n_reservoir))
            readout.append(math.tanh(activation))  # tanh activation

        return {
            "bh_prob": (readout[0] + 1) / 2,      # [-1,1] → [0,1]
            "exit_prob": (readout[1] + 1) / 2,
            "size_scale": max(0.0, readout[2]),
            "regime": max(0, min(2, int((readout[3] + 1) * 1.5))),
            "reservoir_mean_rate": sum(reservoir_state) / len(reservoir_state),
        }

    def _encode_inputs(self, price_change, volume_zscore, bh_mass, beta, cf, hurst) -> list[float]:
        """Encode market state as spike rates (rate coding)."""
        # Normalize each input to [0, 1] approximate range
        return [
            max(0.0, min(1.0, (price_change + 0.05) / 0.10)),     # price change
            max(0.0, min(1.0, (volume_zscore + 3) / 6)),            # volume zscore
            max(0.0, min(1.0, bh_mass / 4.0)),                      # BH mass
            max(0.0, min(1.0, 1.0 - beta)),                          # beta (inverted: low = more TIMELIKE)
            max(0.0, min(1.0, cf / 0.05)),                           # CF
            max(0.0, min(1.0, hurst)),                               # Hurst
        ]

    def _init_sparse_weights(self, n_in, n_out, sparsity) -> list[list[float]]:
        W = [[0.0] * n_out for _ in range(n_in)]
        for i in range(n_in):
            for j in range(n_out):
                if self._rng.random() < sparsity:
                    W[i][j] = self._rng.gauss(0, 1.0 / (n_in * sparsity + 1e-6))
        return W

    def _init_reservoir_weights(self) -> list[list[float]]:
        W = self._init_sparse_weights(
            self.cfg.n_reservoir, self.cfg.n_reservoir, self.cfg.reservoir_sparsity
        )
        # Scale to target spectral radius (power iteration approximation)
        max_abs = max(abs(W[i][j]) for i in range(len(W)) for j in range(len(W[0])))
        if max_abs > 0:
            scale = self.cfg.spectral_radius / max_abs
            W = [[w * scale for w in row] for row in W]
        return W
