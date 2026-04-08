"""
Hawking Radiation Exit Signal (T3-4)
Models BH evaporation using entropy-based thermodynamics.

Physical mapping:
  BH Entropy S = k * ln(Ω), where Ω = order book microstate count
  Hawking Temperature T_H ∝ 1 / BH_mass  (high mass = cold = stable)

When entropy increases rapidly (order book fragmenting) → BH evaporating → exit signal.
Adds Hawking temperature as RL exit state feature.
"""
import math
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

@dataclass
class HawkingConfig:
    entropy_window: int = 10  # bars for entropy rate computation
    temperature_high: float = 2.0  # T_H > this = hot = unstable = exit signal
    evaporation_rate_threshold: float = 0.15  # dS/dt threshold for evaporation detection
    bh_mass_scale: float = 3.5  # nominal BH_MASS_EXTREME for temperature scaling

class HawkingEntropyTracker:
    """
    Tracks BH thermodynamic state for a single instrument.

    Usage:
        tracker = HawkingEntropyTracker()
        result = tracker.update(bh_mass, bid_sizes, ask_sizes, volume)
        if result['evaporating']:
            # Consider exit
    """

    def __init__(self, cfg: HawkingConfig = None):
        self.cfg = cfg or HawkingConfig()
        self._entropy_history: list[float] = []
        self.hawking_temp: float = 0.0
        self.entropy: float = 0.0
        self.evaporation_rate: float = 0.0

    def update(
        self,
        bh_mass: float,
        recent_volumes: list[float],
        bid_depth_proxy: float = 1.0,  # normalized bid side depth
        ask_depth_proxy: float = 1.0,  # normalized ask side depth
    ) -> dict:
        """
        Update thermodynamic state.

        Returns dict with:
          entropy: float — current BH entropy estimate
          hawking_temp: float — evaporation temperature (higher = more unstable)
          evaporation_rate: float — dS/dt (rate of entropy increase)
          evaporating: bool — True if BH is actively evaporating
          exit_signal: float — [0, 1] exit urgency score
        """
        # Compute order book entropy proxy from volume distribution
        # High entropy = many small orders (fragmented) = unstable
        # Low entropy = concentrated large orders = stable
        if recent_volumes and len(recent_volumes) >= 3:
            vol_arr = [max(v, 1e-10) for v in recent_volumes]
            total = sum(vol_arr)
            probs = [v / total for v in vol_arr]
            # Shannon entropy
            raw_entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
            # Normalize by max possible entropy
            max_entropy = math.log(len(probs))
            normalized_entropy = raw_entropy / (max_entropy + 1e-12)
        else:
            normalized_entropy = 0.5  # neutral if no data

        # Modulate by order book imbalance
        imbalance = abs(bid_depth_proxy - ask_depth_proxy) / (bid_depth_proxy + ask_depth_proxy + 1e-12)
        entropy_adjusted = normalized_entropy * (1.0 + 0.3 * imbalance)

        self.entropy = entropy_adjusted
        self._entropy_history.append(entropy_adjusted)
        if len(self._entropy_history) > self.cfg.entropy_window:
            self._entropy_history.pop(0)

        # Hawking temperature: T_H ∝ 1 / mass
        # High mass → low temperature → stable (cold black hole = long-lived)
        # Low mass → high temperature → evaporating rapidly
        safe_mass = max(bh_mass, 0.01)
        self.hawking_temp = self.cfg.bh_mass_scale / safe_mass

        # Evaporation rate: dS/dt
        if len(self._entropy_history) >= 3:
            recent = self._entropy_history[-3:]
            self.evaporation_rate = (recent[-1] - recent[0]) / 2.0  # finite difference
        else:
            self.evaporation_rate = 0.0

        evaporating = (
            self.hawking_temp > self.cfg.temperature_high or
            self.evaporation_rate > self.cfg.evaporation_rate_threshold
        )

        # Exit signal: blend temperature and evaporation rate
        exit_signal = min(1.0, max(0.0,
            0.5 * (self.hawking_temp / (self.cfg.temperature_high * 2)) +
            0.5 * (self.evaporation_rate / (self.cfg.evaporation_rate_threshold * 2))
        ))

        return {
            "entropy": self.entropy,
            "hawking_temp": self.hawking_temp,
            "evaporation_rate": self.evaporation_rate,
            "evaporating": evaporating,
            "exit_signal": exit_signal,
        }

    @property
    def rl_state_feature(self) -> int:
        """
        Maps Hawking temperature to a 5-bin RL state feature (0=cold/stable, 4=hot/evaporating).
        Use this as an additional feature in the RL exit Q-table state.
        """
        T = self.hawking_temp
        if T < 0.5:   return 0
        if T < 1.0:   return 1
        if T < 1.5:   return 2
        if T < 2.0:   return 3
        return 4
