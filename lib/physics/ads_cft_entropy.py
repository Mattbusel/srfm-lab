"""
AdS/CFT-Inspired Financial Entanglement Entropy (T4-6)
Implements the practically usable part: entanglement entropy between instrument subsets.

Physical basis (Ryu-Takayanagi formula):
  S_entanglement(A) = Area(minimal surface in bulk) / (4G)

Financial mapping:
  S_ent(instrument_subset_A | subset_B) = von Neumann entropy of reduced density matrix
  ρ_A = Tr_B[ρ_AB] where ρ_AB is the joint state of the bipartite market

Practically: measures information flow / mutual information between market sectors.
High entanglement entropy between crypto and equity → regime coupling
Low entropy → sectors decoupled (diversification opportunity)
"""
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class AdSCFTConfig:
    window: int = 100          # bars for correlation estimation
    min_history: int = 30      # minimum bars before computing entropy
    entropy_change_threshold: float = 0.3  # significant entropy change

class MarketEntanglementEntropy:
    """
    Computes von Neumann-inspired entanglement entropy between market sectors.

    Usage:
        mee = MarketEntanglementEntropy(
            crypto_syms=["BTC", "ETH", "AVAX"],
            equity_syms=["SPY", "QQQ", "GLD"]
        )
        # Update with each bar's returns
        mee.update_returns({"BTC": 0.012, "ETH": 0.008, "SPY": -0.003, ...})
        result = mee.compute_entanglement()
        print(result["entropy"])  # higher = more coupled
    """

    def __init__(
        self,
        crypto_syms: list[str],
        equity_syms: list[str],
        cfg: AdSCFTConfig = None,
    ):
        self.cfg = cfg or AdSCFTConfig()
        self.crypto_syms = crypto_syms
        self.equity_syms = equity_syms
        self.all_syms = crypto_syms + equity_syms
        self._returns: dict[str, list[float]] = {s: [] for s in self.all_syms}
        self._entropy_history: list[float] = []
        self.current_entropy: float = 0.0
        self.entropy_regime: str = "neutral"

    def update_returns(self, returns: dict[str, float]):
        """Update with one bar of returns. Keys are symbol names."""
        for sym in self.all_syms:
            if sym in returns:
                self._returns[sym].append(returns[sym])
                if len(self._returns[sym]) > self.cfg.window:
                    self._returns[sym].pop(0)

    def compute_entanglement(self) -> dict:
        """
        Compute entanglement entropy between crypto and equity sectors.

        Returns:
          entropy: float — von Neumann entropy of reduced density matrix
          entropy_change: float — delta from last computation
          coupling_regime: str — "high" / "medium" / "low"
          phase_transition: bool — rapid entropy change detected
        """
        # Check we have enough history for all symbols
        min_hist = min(len(v) for v in self._returns.values() if v)
        if min_hist < self.cfg.min_history:
            return {"entropy": 0.0, "entropy_change": 0.0, "coupling_regime": "unknown", "phase_transition": False}

        # Build return matrix: rows = time, cols = symbols
        n_bars = min(self.cfg.window, min_hist)
        data = np.array([self._returns[s][-n_bars:] for s in self.all_syms]).T  # (n_bars, n_syms)

        # Compute joint covariance matrix
        try:
            cov = np.cov(data.T)  # (n_syms, n_syms)
        except Exception:
            return {"entropy": 0.0, "entropy_change": 0.0, "coupling_regime": "unknown", "phase_transition": False}

        # Density matrix analog: normalized covariance matrix
        # ρ_AB = cov / tr(cov)
        trace_cov = np.trace(cov)
        if trace_cov < 1e-12:
            return {"entropy": 0.0, "entropy_change": 0.0, "coupling_regime": "unknown", "phase_transition": False}
        rho_AB = cov / trace_cov

        # Reduced density matrix for crypto sector: ρ_A = Tr_B[ρ_AB]
        n_crypto = len(self.crypto_syms)
        rho_A = rho_AB[:n_crypto, :n_crypto]
        # Renormalize
        trace_A = np.trace(rho_A)
        if trace_A > 1e-12:
            rho_A = rho_A / trace_A

        # Von Neumann entropy: S = -Tr[ρ log ρ] = -Σ λᵢ log λᵢ
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = np.maximum(eigenvalues, 1e-12)  # numerical stability
        eigenvalues = eigenvalues / eigenvalues.sum()  # normalize to sum to 1
        entropy = -float(np.sum(eigenvalues * np.log(eigenvalues + 1e-12)))

        prev_entropy = self._entropy_history[-1] if self._entropy_history else entropy
        entropy_change = entropy - prev_entropy

        self._entropy_history.append(entropy)
        if len(self._entropy_history) > 200:
            self._entropy_history.pop(0)
        self.current_entropy = entropy

        # Regime classification
        max_possible_entropy = math.log(n_crypto)  # ln(n) for uniform distribution
        normalized = entropy / (max_possible_entropy + 1e-12)

        if normalized > 0.7:
            coupling_regime = "high"   # sectors highly coupled → reduce diversification benefit
        elif normalized > 0.4:
            coupling_regime = "medium"
        else:
            coupling_regime = "low"    # sectors decoupled → diversification opportunity
        self.entropy_regime = coupling_regime

        # Phase transition: rapid entropy change
        phase_transition = abs(entropy_change) > self.cfg.entropy_change_threshold

        if phase_transition:
            log.warning("AdS/CFT: market phase transition detected! entropy_change=%.3f", entropy_change)

        return {
            "entropy": entropy,
            "entropy_normalized": normalized,
            "entropy_change": entropy_change,
            "coupling_regime": coupling_regime,
            "phase_transition": phase_transition,
        }
