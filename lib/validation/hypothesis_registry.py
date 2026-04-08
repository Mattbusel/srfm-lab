"""
Hypothesis Registry with Falsification Criteria (A5)
Self-auditing scientific infrastructure for every signal in the system.

For each signal/parameter, maintains:
  - The hypothesis (formal statement)
  - Falsification criteria (what would prove it wrong)
  - Current test statistic
  - Pass/fail status

Generates regular falsification reports.
"""
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger(__name__)

@dataclass
class Hypothesis:
    name: str
    signal_id: str                # e.g. "bh_physics", "garch_vol", "ou_reversion"
    statement: str                # formal hypothesis text
    falsification_criterion: str  # what would falsify it
    min_obs: int = 90             # minimum observations before evaluating
    significance_level: float = 0.48  # H₀ reject threshold (P(positive) < this)

    # Runtime state
    observations: list = field(default_factory=list)  # (is_positive: bool) per trade
    created_at: float = field(default_factory=time.time)
    last_evaluated: float = 0.0
    current_p_positive: float = 0.5
    falsified: bool = False
    falsified_at: Optional[float] = None
    test_statistic: float = 0.0

class HypothesisRegistry:
    """
    Registry of all system hypotheses with automatic falsification testing.

    Pre-registered hypotheses cover all major SRFM signals.

    Usage:
        registry = HypothesisRegistry()
        registry.record_outcome("bh_physics", is_profitable=True)
        report = registry.evaluate_all()
        falsified = registry.get_falsified()
    """

    # Pre-defined hypotheses for all major system signals
    BUILT_IN_HYPOTHESES = [
        Hypothesis(
            name="BH Physics Entry",
            signal_id="bh_physics",
            statement="BH mass > BH_FORM threshold predicts positive forward returns",
            falsification_criterion="P(positive return | BH > 1.92) < 0.48 over 90 trading days",
            min_obs=90, significance_level=0.48,
        ),
        Hypothesis(
            name="GARCH Volatility Scaling",
            signal_id="garch_vol",
            statement="Scaling position size inversely with GARCH conditional variance improves risk-adjusted returns",
            falsification_criterion="Sharpe(GARCH-scaled) < Sharpe(uniform-size) over 120 days",
            min_obs=120, significance_level=0.48,
        ),
        Hypothesis(
            name="OU Mean Reversion Dampener",
            signal_id="ou_reversion",
            statement="OU zscore dampening reduces false positive BH signals in mean-reverting regimes",
            falsification_criterion="Win rate with OU dampening < Win rate without over 90 days",
            min_obs=90, significance_level=0.48,
        ),
        Hypothesis(
            name="Hurst Regime Filter",
            signal_id="hurst_filter",
            statement="Hurst-conditional signal weighting improves performance vs uniform weights",
            falsification_criterion="Sharpe(Hurst-weighted) < Sharpe(uniform) over 60 days",
            min_obs=60, significance_level=0.48,
        ),
        Hypothesis(
            name="RL Exit Optimizer",
            signal_id="rl_exit",
            statement="RL Q-table exit policy produces better hold times and reduced loss exits vs fixed holds",
            falsification_criterion="Avg hold < 4 bars or win rate < 0.40 over 60 days",
            min_obs=60, significance_level=0.40,
        ),
        Hypothesis(
            name="QuatNav Spin Rate",
            signal_id="quatnav_spin",
            statement="Quaternion spin rate > 0.02 is a leading indicator of BH formation",
            falsification_criterion="P(BH within 5 bars | spin_rate > 0.02) < 0.55",
            min_obs=100, significance_level=0.55,
        ),
        Hypothesis(
            name="Granger BTC Lead",
            signal_id="granger_btc",
            statement="BTC Granger-causes altcoin moves with |corr|>0.3; boosting on this signal improves alt Sharpe",
            falsification_criterion="Alt Sharpe with Granger boost < Alt Sharpe without over 90 days",
            min_obs=90, significance_level=0.48,
        ),
        Hypothesis(
            name="Multi-TF Coherence Filter",
            signal_id="mtf_coherence",
            statement="Requiring 2+ timeframe BH coherence improves win rate vs single-TF entries",
            falsification_criterion="Win rate (2+ TF) < Win rate (1 TF) over 60 days",
            min_obs=60, significance_level=0.48,
        ),
        Hypothesis(
            name="Gravitational Wave Sizing",
            signal_id="grav_wave",
            statement="Coordinated 2+ instrument BH formations (within 8 bars) warrant 1.5x sizing",
            falsification_criterion="Sharpe(grav wave 1.5x) < Sharpe(1x) over 90 days",
            min_obs=90, significance_level=0.48,
        ),
        Hypothesis(
            name="GARCH Volatility Hard Gate",
            signal_id="garch_gate",
            statement="Blocking trades when GARCH var > 3x median reduces max drawdown",
            falsification_criterion="Max drawdown (gated) > Max drawdown (ungated) over 90 days",
            min_obs=90, significance_level=0.48,
        ),
    ]

    def __init__(self, save_path: Optional[str] = None):
        self._hypotheses: dict[str, Hypothesis] = {}
        self._save_path = Path(save_path) if save_path else None

        # Register built-in hypotheses
        for h in self.BUILT_IN_HYPOTHESES:
            self.register(h)

    def register(self, hypothesis: Hypothesis):
        """Register a new hypothesis."""
        self._hypotheses[hypothesis.signal_id] = hypothesis
        log.debug("HypothesisRegistry: registered '%s'", hypothesis.name)

    def record_outcome(self, signal_id: str, is_positive: bool, metadata: dict = None):
        """
        Record a trade outcome for a specific signal.
        is_positive: True if the trade was profitable.
        """
        h = self._hypotheses.get(signal_id)
        if h is None:
            return
        h.observations.append(is_positive)
        if len(h.observations) > 2000:
            h.observations = h.observations[-2000:]

    def evaluate_all(self) -> dict:
        """
        Evaluate all hypotheses and return status report.
        Returns dict: signal_id → {status, p_positive, n_obs, falsified}
        """
        results = {}
        for sid, h in self._hypotheses.items():
            n = len(h.observations)
            if n < h.min_obs:
                results[sid] = {
                    "status": "insufficient_data",
                    "n_obs": n, "needed": h.min_obs,
                    "falsified": False,
                }
                continue

            n_positive = sum(h.observations)
            p_pos = n_positive / n

            # Binomial test: is P(positive) significantly below falsification level?
            # Using Wilson score interval lower bound
            z = 1.645  # 95% one-sided
            center = (p_pos + z*z/(2*n)) / (1 + z*z/n)
            margin = z * math.sqrt(p_pos*(1-p_pos)/n + z*z/(4*n*n)) / (1 + z*z/n)
            lower_bound = center - margin

            newly_falsified = lower_bound < h.significance_level and n >= h.min_obs

            h.current_p_positive = p_pos
            h.test_statistic = lower_bound
            h.last_evaluated = time.time()

            if newly_falsified and not h.falsified:
                h.falsified = True
                h.falsified_at = time.time()
                log.warning(
                    "HYPOTHESIS FALSIFIED: '%s' (signal=%s) — P(positive)=%.3f < threshold=%.3f",
                    h.name, sid, p_pos, h.significance_level
                )

            status = "FALSIFIED" if h.falsified else ("SUPPORTED" if p_pos > h.significance_level + 0.05 else "BORDERLINE")
            results[sid] = {
                "name": h.name,
                "status": status,
                "p_positive": p_pos,
                "lower_bound": lower_bound,
                "n_obs": n,
                "falsified": h.falsified,
                "statement": h.statement,
                "criterion": h.falsification_criterion,
            }

        return results

    def get_falsified(self) -> list[str]:
        """Return list of falsified signal IDs."""
        return [sid for sid, h in self._hypotheses.items() if h.falsified]

    def save(self):
        """Persist registry state to JSON."""
        if not self._save_path:
            return
        state = {}
        for sid, h in self._hypotheses.items():
            state[sid] = {
                "name": h.name,
                "observations_count": len(h.observations),
                "p_positive": h.current_p_positive,
                "falsified": h.falsified,
                "n_obs": len(h.observations),
            }
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._save_path, "w") as f:
            json.dump(state, f, indent=2)
