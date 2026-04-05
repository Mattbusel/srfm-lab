"""
debate-system/agents/risk_manager.py

RiskManagementAnalyst: focuses on tail risk, drawdown, and portfolio
concentration effects of implementing a hypothesis.

Hard veto power
---------------
This agent has the unique ability to issue a hard AGAINST vote that the
chamber records as a veto — if implementing the hypothesis would increase
max drawdown by more than 20 percentage points, it is rejected regardless
of all other votes.  This prevents the portfolio from taking on catastrophic
tail risk even when a strategy looks statistically sound.

Rationale: max drawdown is non-negotiable in live trading.  A 40% drawdown
forces position reduction at exactly the wrong time (forced selling into
weakness), creates psychological pressure to abandon a strategy that may
recover, and can trigger fund redemptions / margin calls.

Risk assessment dimensions
--------------------------
1. Max drawdown impact: would this increase portfolio MDD by > 20%?
2. Worst 5-day outcome: what happens if every trade in 5 days goes wrong?
3. Concentration risk: does this create a cluster of correlated exposures?
4. Diversification impact: does this reduce or increase daily return variance?
5. Leverage implicitly embedded: does the hypothesis require sizing up?
"""

from __future__ import annotations

from typing import Any

from hypothesis.types import Hypothesis
from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote


class RiskManagementAnalyst(BaseAnalyst):
    """
    Portfolio risk and tail-event specialist.  Has hard veto on drawdown.
    """

    # Hard veto thresholds
    MAX_ALLOWABLE_MDD_INCREASE = 0.20    # 20 percentage points
    SOFT_MDD_CONCERN = 0.10              # 10pp — raise concern, not veto

    # Worst-5-day scenario threshold (as fraction of portfolio NAV)
    MAX_5DAY_LOSS = 0.15                 # 15% portfolio loss in worst 5 days

    # Concentration: max allowed share of correlated assets
    MAX_CONCENTRATION_RATIO = 0.40       # 40% of book in same risk factor

    def __init__(self) -> None:
        super().__init__(
            name="RiskManagementAnalyst",
            specialization=(
                "Max drawdown, tail risk, concentration risk, "
                "diversification, worst-case scenario analysis"
            ),
        )

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: dict[str, Any],
    ) -> AnalystVerdict:
        """
        Evaluate the tail risk of implementing a hypothesis.

        Expected market_data keys
        -------------------------
        current_mdd             : float - current portfolio max drawdown (positive)
        hypothesis_mdd          : float - projected MDD with hypothesis active
        worst_5day_pnl          : float - worst observed 5-day outcome (negative)
        portfolio_var_change    : float - % change in daily VaR (positive = worse)
        concentration_ratio     : float - fraction of book in hypothesis's risk factor
        diversification_score   : float - 0-1, 1 = fully diversifying
        hypothesis_n_positions  : int   - number of simultaneous positions added
        avg_position_size_pct   : float - average position size as % of portfolio
        """
        current_mdd: float = float(market_data.get("current_mdd", 0.0))
        hyp_mdd: float = float(market_data.get("hypothesis_mdd", 0.0))
        worst_5day: float = float(market_data.get("worst_5day_pnl", 0.0))
        var_change: float = float(market_data.get("portfolio_var_change", 0.0))
        concentration: float = float(market_data.get("concentration_ratio", 0.0))
        diversification: float = float(market_data.get("diversification_score", 0.5))
        n_positions: int = int(market_data.get("hypothesis_n_positions", 1))
        avg_pos_size: float = float(market_data.get("avg_position_size_pct", 0.02))

        concerns: list[str] = []
        veto = False

        # --- Hard veto: max drawdown increase ---------------------------
        mdd_increase = hyp_mdd - current_mdd
        if mdd_increase > self.MAX_ALLOWABLE_MDD_INCREASE:
            veto = True
            concerns.append(
                f"HARD VETO: hypothesis increases max drawdown by "
                f"{mdd_increase:.1%} (current={current_mdd:.1%}, "
                f"projected={hyp_mdd:.1%}). Exceeds 20pp hard limit. "
                f"This risk is non-negotiable — drawdown at this level "
                f"forces position liquidation at worst possible time."
            )

        elif mdd_increase > self.SOFT_MDD_CONCERN:
            concerns.append(
                f"Drawdown concern: MDD increases by {mdd_increase:.1%} "
                f"(from {current_mdd:.1%} to {hyp_mdd:.1%}). "
                f"Approaching hard veto threshold."
            )

        # --- Worst 5-day scenario ---------------------------------------
        if worst_5day < -self.MAX_5DAY_LOSS:
            concerns.append(
                f"Severe tail risk: worst 5-day outcome is {worst_5day:.1%} of "
                f"portfolio NAV. A concentrated bad week could trigger risk limits "
                f"and force premature strategy shutdown."
            )
            if not veto:
                veto = abs(worst_5day) > 0.30   # 30% in 5 days is existential

        # --- Concentration risk -----------------------------------------
        if concentration > self.MAX_CONCENTRATION_RATIO:
            concerns.append(
                f"Concentration risk: {concentration:.0%} of book would be "
                f"exposed to the same risk factor. Single-factor drawdown "
                f"would be catastrophic."
            )

        # --- VaR impact -------------------------------------------------
        if var_change > 0.15:
            concerns.append(
                f"Daily VaR increases by {var_change:.0%}. "
                f"Portfolio tail distribution becomes significantly fatter."
            )

        # --- Diversification assessment ---------------------------------
        if diversification < 0.3:
            concerns.append(
                f"Low diversification score ({diversification:.2f}). "
                f"Hypothesis adds correlated risk, not independent alpha."
            )

        # --- Size sanity check ------------------------------------------
        total_exposure = n_positions * avg_pos_size
        if total_exposure > 0.25:
            concerns.append(
                f"Implied total exposure {total_exposure:.0%} of portfolio "
                f"({n_positions} positions × {avg_pos_size:.0%} avg) is large."
            )

        # --- Render verdict ---------------------------------------------
        if veto:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.99,   # near-certain on hard veto
                reasoning=(
                    "RISK VETO ISSUED. " + concerns[0]
                ),
                key_concerns=concerns,
            )

        # Score-based verdict
        risk_score = len([c for c in concerns if "concentration" in c or "VaR" in c or "tail" in c])

        if not concerns:
            return self._make_verdict(
                Vote.FOR,
                confidence=0.80,
                reasoning=(
                    f"Risk profile acceptable. MDD increase={mdd_increase:.1%}, "
                    f"diversification={diversification:.2f}, "
                    f"no concentration or tail-risk concerns."
                ),
                key_concerns=[],
            )
        elif len(concerns) <= 2 and mdd_increase <= self.SOFT_MDD_CONCERN:
            return self._make_verdict(
                Vote.FOR,
                confidence=0.55,
                reasoning=(
                    f"Marginal risk profile. {len(concerns)} concern(s) noted "
                    f"but none hit hard limits. Monitor closely post-deployment."
                ),
                key_concerns=concerns,
            )
        else:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.70 + 0.05 * len(concerns),
                reasoning=(
                    f"Unacceptable risk profile: {len(concerns)} risk concern(s). "
                    f"MDD increase {mdd_increase:.1%}, concentration {concentration:.0%}."
                ),
                key_concerns=concerns,
            )
