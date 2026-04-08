"""
SHAP Explainability Layer (A3)
Every trading decision generates a structured explanation with signal contributions.

Uses simplified SHAP-like attribution (no external shap library dependency):
  - Physics signals (BH, curvature) get linear attribution from signal magnitudes
  - ML signals get feature importance from the XGBoost model's internal attributions
  - RL signals get Q-value decomposition

Outputs human-readable explanation + structured dict for IAE miners.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class SignalContribution:
    signal_name: str
    raw_value: float     # raw signal value
    attribution: float   # contribution to final decision (positive = toward entry)
    weight: float        # signal weight in combination
    description: str     # human-readable explanation of why this signal fired

@dataclass
class TradeExplanation:
    symbol: str
    bar_seq: int
    decision: str  # "ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"
    final_score: float
    dominant_signal: str
    contributions: list[SignalContribution]
    explanation_text: str
    is_profitable: Optional[bool] = None  # filled in post-trade

class SHAPAttributionEngine:
    """
    Generates structured explanations for every trading decision.

    Usage:
        engine = SHAPAttributionEngine()
        explanation = engine.explain_entry(
            symbol="BTC",
            bar_seq=12345,
            bh_mass=2.3, bh_active=True,
            garch_vol=0.025, hurst=0.63,
            ou_zscore=-0.5, ml_score=0.45,
            rl_exit_q_hold=0.08, rl_exit_q_exit=-0.03,
            final_target_frac=0.15,
        )
        print(explanation.explanation_text)
    """

    def __init__(self):
        self._history: list[TradeExplanation] = []

    def explain_entry(
        self,
        symbol: str,
        bar_seq: int,
        bh_mass: float = 0.0,
        bh_active: bool = False,
        garch_vol: float = 0.02,
        hurst: float = 0.5,
        ou_zscore: float = 0.0,
        ml_score: float = 0.0,
        rl_exit_q_hold: float = 0.0,
        rl_exit_q_exit: float = 0.0,
        granger_boost: float = 0.0,
        quatnav_boost: float = 0.0,
        corr_boost: float = 0.0,
        hawking_temp: float = 0.0,
        final_target_frac: float = 0.0,
        **kwargs,
    ) -> TradeExplanation:
        """Generate explanation for an entry decision."""

        contributions = []

        # BH Physics: central signal
        bh_attr = (bh_mass / 3.84) * (1.0 if bh_active else 0.3)  # 3.84 = 2x BH_FORM
        contributions.append(SignalContribution(
            signal_name="bh_physics",
            raw_value=bh_mass,
            attribution=bh_attr,
            weight=1.0,
            description=f"BH mass={bh_mass:.2f} ({'active' if bh_active else 'inactive'})",
        ))

        # GARCH volatility (sizing signal)
        garch_attr = -min(0.3, max(-0.3, (garch_vol - 0.02) * 10))
        contributions.append(SignalContribution(
            signal_name="garch_vol",
            raw_value=garch_vol,
            attribution=garch_attr,
            weight=0.6,
            description=f"GARCH vol={garch_vol:.3f} ({'high' if garch_vol > 0.03 else 'normal'})",
        ))

        # Hurst regime
        if hurst > 0.58:
            hurst_attr = 0.15 * (hurst - 0.58) / 0.22
            hurst_desc = f"Hurst={hurst:.2f} (trending -> amplify BH)"
        elif hurst < 0.42:
            hurst_attr = -0.10  # mean-reverting dampens momentum
            hurst_desc = f"Hurst={hurst:.2f} (mean-reverting -> dampen)"
        else:
            hurst_attr = 0.0
            hurst_desc = f"Hurst={hurst:.2f} (neutral)"
        contributions.append(SignalContribution(
            signal_name="hurst",
            raw_value=hurst,
            attribution=hurst_attr,
            weight=0.5,
            description=hurst_desc,
        ))

        # OU mean reversion
        ou_attr = -abs(ou_zscore) * 0.05 if abs(ou_zscore) > 1.5 else 0.0
        contributions.append(SignalContribution(
            signal_name="ou_reversion",
            raw_value=ou_zscore,
            attribution=ou_attr,
            weight=0.4,
            description=f"OU zscore={ou_zscore:.2f}",
        ))

        # ML signal
        ml_attr = ml_score * 0.3 if abs(ml_score) > 0.1 else 0.0
        contributions.append(SignalContribution(
            signal_name="ml_boost",
            raw_value=ml_score,
            attribution=ml_attr,
            weight=0.5,
            description=f"ML score={ml_score:.3f} ({'boost' if ml_score > 0.3 else 'suppress' if ml_score < -0.3 else 'neutral'})",
        ))

        # RL exit signal
        rl_attr = (rl_exit_q_hold - rl_exit_q_exit) * 0.2
        contributions.append(SignalContribution(
            signal_name="rl_exit",
            raw_value=rl_exit_q_hold - rl_exit_q_exit,
            attribution=rl_attr,
            weight=0.4,
            description=f"RL: hold_q={rl_exit_q_hold:.3f}, exit_q={rl_exit_q_exit:.3f}",
        ))

        # QuatNav boost
        if abs(quatnav_boost) > 0.01:
            contributions.append(SignalContribution(
                signal_name="quatnav",
                raw_value=quatnav_boost,
                attribution=quatnav_boost * 0.5,
                weight=0.3,
                description=f"QuatNav boost={quatnav_boost:.3f}",
            ))

        # Granger/correlation boost
        if abs(granger_boost + corr_boost) > 0.01:
            combined_boost = granger_boost + corr_boost
            contributions.append(SignalContribution(
                signal_name="cross_asset",
                raw_value=combined_boost,
                attribution=combined_boost * 0.3,
                weight=0.4,
                description=f"Cross-asset boost={combined_boost:.3f} (Granger={granger_boost:.2f}, corr={corr_boost:.2f})",
            ))

        # Total attribution
        total_attr = sum(c.attribution * c.weight for c in contributions)

        # Dominant signal
        dominant = max(contributions, key=lambda c: abs(c.attribution * c.weight))

        # Build explanation text
        top_3 = sorted(contributions, key=lambda c: abs(c.attribution), reverse=True)[:3]
        top_strs = [f"{c.signal_name}={c.attribution:+.3f}" for c in top_3]
        decision = "ENTER_LONG" if final_target_frac > 0.01 else "HOLD"

        explanation_text = (
            f"{symbol} {decision}: score={total_attr:+.3f} | "
            f"dominant={dominant.signal_name} | "
            f"top signals: {', '.join(top_strs)}"
        )

        explanation = TradeExplanation(
            symbol=symbol,
            bar_seq=bar_seq,
            decision=decision,
            final_score=total_attr,
            dominant_signal=dominant.signal_name,
            contributions=contributions,
            explanation_text=explanation_text,
        )

        self._history.append(explanation)
        if len(self._history) > 10000:
            self._history = self._history[-10000:]

        return explanation

    def mark_outcome(self, symbol: str, bar_seq: int, is_profitable: bool):
        """Record trade outcome for a previously explained decision."""
        for exp in reversed(self._history):
            if exp.symbol == symbol and exp.bar_seq == bar_seq:
                exp.is_profitable = is_profitable
                break

    def get_signal_win_rates(self) -> dict[str, float]:
        """
        Compute win rate by dominant signal. Useful for IAE miners.
        Returns {signal_name: win_rate} for decisions with known outcomes.
        """
        signal_wins: dict[str, list[bool]] = {}
        for exp in self._history:
            if exp.is_profitable is None:
                continue
            if exp.dominant_signal not in signal_wins:
                signal_wins[exp.dominant_signal] = []
            signal_wins[exp.dominant_signal].append(exp.is_profitable)

        return {
            sig: sum(outcomes) / len(outcomes)
            for sig, outcomes in signal_wins.items()
            if outcomes
        }
