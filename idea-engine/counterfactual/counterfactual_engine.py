"""
Counterfactual reasoning engine — "what would have happened if...?"

Implements:
  - Counterfactual trade analysis: what if we had entered earlier/later?
  - Parameter sensitivity: what if the lookback was X instead of Y?
  - Regime counterfactual: what if the regime had been different?
  - Portfolio counterfactual: what if allocation was different?
  - Event removal: what if a specific event (news, earnings) hadn't happened?
  - Synthetic counterfactual via causal inference (potential outcomes)
  - Counterfactual P&L: reconstruct alternate histories
  - Scenario comparison: actual vs counterfactual equity curves
  - Regret analysis: which decisions had highest opportunity cost?
  - Learning: distill counterfactual insights into future decision rules
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


# ── Counterfactual Result ─────────────────────────────────────────────────────

@dataclass
class CounterfactualResult:
    """Result of a counterfactual analysis."""
    name: str
    description: str
    actual_pnl: float
    counterfactual_pnl: float
    opportunity_cost: float          # CF - actual (positive = missed opportunity)
    actual_sharpe: float
    counterfactual_sharpe: float
    actual_max_dd: float
    counterfactual_max_dd: float
    actual_equity_curve: np.ndarray
    counterfactual_equity_curve: np.ndarray
    insight: str                     # key takeaway
    confidence: float                # 0-1 how reliable is this counterfactual


# ── Counterfactual Engine ─────────────────────────────────────────────────────

class CounterfactualEngine:
    """Engine for counterfactual trade analysis."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _equity_curve(self, returns: np.ndarray) -> np.ndarray:
        return np.cumprod(1 + returns)

    def _sharpe(self, returns: np.ndarray) -> float:
        if len(returns) < 5 or returns.std() < 1e-10:
            return 0.0
        return float(returns.mean() / returns.std() * math.sqrt(252))

    def _max_dd(self, returns: np.ndarray) -> float:
        eq = self._equity_curve(returns)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-10)
        return float(dd.max())

    # ── Trade Timing Counterfactual ───────────────────────────────────────

    def timing_counterfactual(
        self,
        signal: np.ndarray,
        returns: np.ndarray,
        actual_entry_idx: int,
        actual_exit_idx: int,
        offsets: list[int] = None,
    ) -> list[CounterfactualResult]:
        """What if we entered/exited at different times?"""
        if offsets is None:
            offsets = [-5, -3, -1, 1, 3, 5]

        n = len(returns)
        actual_ret = returns[actual_entry_idx:actual_exit_idx]
        actual_pnl = float(np.sum(actual_ret))
        actual_sharpe = self._sharpe(actual_ret)
        actual_dd = self._max_dd(actual_ret)
        actual_eq = self._equity_curve(actual_ret)

        results = []
        for offset in offsets:
            cf_entry = max(0, actual_entry_idx + offset)
            cf_exit = min(n, actual_exit_idx + offset)
            if cf_entry >= cf_exit:
                continue

            cf_ret = returns[cf_entry:cf_exit]
            cf_pnl = float(np.sum(cf_ret))
            cf_sharpe = self._sharpe(cf_ret)
            cf_dd = self._max_dd(cf_ret)
            cf_eq = self._equity_curve(cf_ret)

            insight = (
                f"{'Earlier' if offset < 0 else 'Later'} by {abs(offset)} days: "
                f"PnL {'better' if cf_pnl > actual_pnl else 'worse'} by "
                f"{abs(cf_pnl - actual_pnl)*100:.1f}%"
            )

            results.append(CounterfactualResult(
                name=f"timing_offset_{offset:+d}",
                description=f"Entry/exit shifted by {offset} days",
                actual_pnl=actual_pnl,
                counterfactual_pnl=cf_pnl,
                opportunity_cost=cf_pnl - actual_pnl,
                actual_sharpe=actual_sharpe,
                counterfactual_sharpe=cf_sharpe,
                actual_max_dd=actual_dd,
                counterfactual_max_dd=cf_dd,
                actual_equity_curve=actual_eq,
                counterfactual_equity_curve=cf_eq,
                insight=insight,
                confidence=0.9,
            ))

        return results

    # ── Parameter Sensitivity Counterfactual ──────────────────────────────

    def parameter_counterfactual(
        self,
        signal_generator: Callable[[dict], np.ndarray],
        base_params: dict,
        param_name: str,
        param_values: list,
        returns: np.ndarray,
    ) -> list[CounterfactualResult]:
        """What if we used different parameter values?"""
        # Actual with base params
        actual_signal = signal_generator(base_params)
        actual_strat_ret = actual_signal * returns
        actual_pnl = float(np.sum(actual_strat_ret))
        actual_sharpe = self._sharpe(actual_strat_ret)
        actual_dd = self._max_dd(actual_strat_ret)
        actual_eq = self._equity_curve(actual_strat_ret)

        results = []
        for val in param_values:
            cf_params = {**base_params, param_name: val}
            cf_signal = signal_generator(cf_params)
            cf_ret = cf_signal * returns
            cf_pnl = float(np.sum(cf_ret))

            results.append(CounterfactualResult(
                name=f"{param_name}={val}",
                description=f"What if {param_name} was {val} instead of {base_params.get(param_name)}?",
                actual_pnl=actual_pnl,
                counterfactual_pnl=cf_pnl,
                opportunity_cost=cf_pnl - actual_pnl,
                actual_sharpe=actual_sharpe,
                counterfactual_sharpe=self._sharpe(cf_ret),
                actual_max_dd=actual_dd,
                counterfactual_max_dd=self._max_dd(cf_ret),
                actual_equity_curve=actual_eq,
                counterfactual_equity_curve=self._equity_curve(cf_ret),
                insight=f"{param_name}={val}: Sharpe {self._sharpe(cf_ret):.2f} vs actual {actual_sharpe:.2f}",
                confidence=0.85,
            ))

        return results

    # ── Allocation Counterfactual ─────────────────────────────────────────

    def allocation_counterfactual(
        self,
        actual_weights: np.ndarray,
        alternative_weights: list[tuple[str, np.ndarray]],
        asset_returns: np.ndarray,  # (T, N)
    ) -> list[CounterfactualResult]:
        """What if allocation had been different?"""
        actual_ret = asset_returns @ actual_weights
        actual_pnl = float(np.sum(actual_ret))
        actual_sharpe = self._sharpe(actual_ret)
        actual_dd = self._max_dd(actual_ret)
        actual_eq = self._equity_curve(actual_ret)

        results = []
        for name, alt_w in alternative_weights:
            cf_ret = asset_returns @ alt_w
            cf_pnl = float(np.sum(cf_ret))
            cf_sharpe = self._sharpe(cf_ret)

            insight = f"{name}: Sharpe {cf_sharpe:.2f} vs actual {actual_sharpe:.2f}"
            if cf_sharpe > actual_sharpe:
                insight += " — BETTER risk-adjusted return"
            else:
                insight += " — worse risk-adjusted return"

            results.append(CounterfactualResult(
                name=f"alloc_{name}",
                description=f"Counterfactual allocation: {name}",
                actual_pnl=actual_pnl,
                counterfactual_pnl=cf_pnl,
                opportunity_cost=cf_pnl - actual_pnl,
                actual_sharpe=actual_sharpe,
                counterfactual_sharpe=cf_sharpe,
                actual_max_dd=actual_dd,
                counterfactual_max_dd=self._max_dd(cf_ret),
                actual_equity_curve=actual_eq,
                counterfactual_equity_curve=self._equity_curve(cf_ret),
                insight=insight,
                confidence=0.95,
            ))

        return results

    # ── Regime Counterfactual ─────────────────────────────────────────────

    def regime_counterfactual(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray,
        strategy_returns: np.ndarray,
        hypothetical_regime: str,
        regime_return_distributions: dict[str, tuple[float, float]],  # regime -> (mean, std)
    ) -> CounterfactualResult:
        """What if the regime had been different?"""
        actual_ret = strategy_returns.copy()
        actual_pnl = float(np.sum(actual_ret))
        actual_sharpe = self._sharpe(actual_ret)

        # Generate counterfactual returns from hypothetical regime
        mu, sigma = regime_return_distributions.get(hypothetical_regime, (0.0, 0.01))
        n = len(returns)
        cf_market = self.rng.normal(mu / 252, sigma / math.sqrt(252), n)

        # Counterfactual strategy: same signal applied to different market
        signal_ratio = strategy_returns / (returns + 1e-10)
        signal_ratio = np.clip(signal_ratio, -5, 5)
        cf_strat = signal_ratio * cf_market
        cf_pnl = float(np.sum(cf_strat))
        cf_sharpe = self._sharpe(cf_strat)

        return CounterfactualResult(
            name=f"regime_{hypothetical_regime}",
            description=f"What if regime was {hypothetical_regime}?",
            actual_pnl=actual_pnl,
            counterfactual_pnl=cf_pnl,
            opportunity_cost=cf_pnl - actual_pnl,
            actual_sharpe=actual_sharpe,
            counterfactual_sharpe=cf_sharpe,
            actual_max_dd=self._max_dd(actual_ret),
            counterfactual_max_dd=self._max_dd(cf_strat),
            actual_equity_curve=self._equity_curve(actual_ret),
            counterfactual_equity_curve=self._equity_curve(cf_strat),
            insight=f"In {hypothetical_regime} regime: Sharpe would be {cf_sharpe:.2f}",
            confidence=0.5,  # lower confidence — hypothetical
        )

    # ── Event Removal ─────────────────────────────────────────────────────

    def event_removal_counterfactual(
        self,
        returns: np.ndarray,
        event_dates: list[int],     # indices of event days
        window: int = 3,            # days around event to replace
    ) -> CounterfactualResult:
        """What if specific events hadn't happened?"""
        cf_returns = returns.copy()

        # Replace event windows with average non-event returns
        event_mask = np.zeros(len(returns), dtype=bool)
        for idx in event_dates:
            for d in range(max(0, idx - window), min(len(returns), idx + window + 1)):
                event_mask[d] = True

        non_event_mean = float(returns[~event_mask].mean())
        non_event_std = float(returns[~event_mask].std())
        cf_returns[event_mask] = self.rng.normal(non_event_mean, non_event_std, int(event_mask.sum()))

        actual_pnl = float(np.sum(returns))
        cf_pnl = float(np.sum(cf_returns))
        event_pnl = float(np.sum(returns[event_mask]))

        return CounterfactualResult(
            name="event_removal",
            description=f"Removed {len(event_dates)} events ({int(event_mask.sum())} days)",
            actual_pnl=actual_pnl,
            counterfactual_pnl=cf_pnl,
            opportunity_cost=cf_pnl - actual_pnl,
            actual_sharpe=self._sharpe(returns),
            counterfactual_sharpe=self._sharpe(cf_returns),
            actual_max_dd=self._max_dd(returns),
            counterfactual_max_dd=self._max_dd(cf_returns),
            actual_equity_curve=self._equity_curve(returns),
            counterfactual_equity_curve=self._equity_curve(cf_returns),
            insight=f"Events contributed {event_pnl*100:.1f}% to total PnL ({event_pnl/max(abs(actual_pnl), 1e-10)*100:.0f}% of total)",
            confidence=0.7,
        )

    # ── Regret Analysis ───────────────────────────────────────────────────

    def regret_analysis(
        self,
        decisions: list[dict],  # each: {'name', 'actual_return', 'best_alternative_return'}
    ) -> dict:
        """Compute regret metrics across a series of decisions."""
        if not decisions:
            return {"total_regret": 0.0}

        regrets = []
        for d in decisions:
            actual = d.get("actual_return", 0)
            best = d.get("best_alternative_return", 0)
            regret = max(best - actual, 0)
            regrets.append({
                "name": d.get("name", ""),
                "actual": actual,
                "best_alternative": best,
                "regret": regret,
            })

        regret_values = [r["regret"] for r in regrets]
        total_regret = float(sum(regret_values))
        avg_regret = float(np.mean(regret_values))

        # Hindsight optimal
        optimal_total = float(sum(d.get("best_alternative_return", 0) for d in decisions))
        actual_total = float(sum(d.get("actual_return", 0) for d in decisions))

        # Worst decisions
        regrets.sort(key=lambda x: x["regret"], reverse=True)

        return {
            "total_regret": total_regret,
            "avg_regret_per_decision": avg_regret,
            "actual_total_return": actual_total,
            "hindsight_optimal_return": optimal_total,
            "regret_as_pct_of_optimal": float(total_regret / max(abs(optimal_total), 1e-10) * 100),
            "worst_decisions": regrets[:5],
            "n_decisions": len(decisions),
            "n_regretful": int(sum(1 for r in regret_values if r > 0)),
        }

    # ── Synthetic Counterfactual (Potential Outcomes) ─────────────────────

    def synthetic_control_counterfactual(
        self,
        treated_returns: np.ndarray,     # returns of treated (actual) unit
        control_returns: np.ndarray,     # (T, N) returns of control pool
        treatment_start: int,             # when treatment (trade) began
    ) -> CounterfactualResult:
        """
        Abadie-style synthetic control: what would treated unit have done
        without the treatment (trade/strategy)?
        """
        T, N = control_returns.shape
        pre_treatment = slice(0, treatment_start)

        # Fit weights on pre-treatment period
        y_pre = treated_returns[pre_treatment]
        X_pre = control_returns[pre_treatment]

        # Constrained regression: weights sum to 1, non-negative
        w = np.ones(N) / N
        for _ in range(200):
            resid = y_pre - X_pre @ w
            grad = -2 * X_pre.T @ resid / len(y_pre)
            w -= 0.01 * grad
            w = np.maximum(w, 0)
            w /= w.sum() + 1e-10

        # Counterfactual: synthetic control post-treatment
        synthetic_full = control_returns @ w

        actual_post = treated_returns[treatment_start:]
        cf_post = synthetic_full[treatment_start:]

        # Treatment effect
        effect = actual_post - cf_post
        cum_effect = float(np.sum(effect))

        return CounterfactualResult(
            name="synthetic_control",
            description=f"Synthetic control counterfactual from {N} control units",
            actual_pnl=float(np.sum(actual_post)),
            counterfactual_pnl=float(np.sum(cf_post)),
            opportunity_cost=float(-cum_effect),  # negative = actual was better
            actual_sharpe=self._sharpe(actual_post),
            counterfactual_sharpe=self._sharpe(cf_post),
            actual_max_dd=self._max_dd(actual_post),
            counterfactual_max_dd=self._max_dd(cf_post),
            actual_equity_curve=self._equity_curve(treated_returns),
            counterfactual_equity_curve=self._equity_curve(synthetic_full),
            insight=f"Treatment effect: {cum_effect*100:.2f}% cumulative",
            confidence=0.6,
        )

    # ── Learning from Counterfactuals ─────────────────────────────────────

    def distill_insights(self, results: list[CounterfactualResult]) -> dict:
        """Extract actionable insights from counterfactual analyses."""
        if not results:
            return {"n_analyses": 0}

        # Best and worst counterfactuals
        by_opp_cost = sorted(results, key=lambda r: r.opportunity_cost, reverse=True)

        # Patterns
        timing_better = sum(1 for r in results if "timing" in r.name and r.opportunity_cost > 0)
        timing_total = sum(1 for r in results if "timing" in r.name)
        alloc_better = sum(1 for r in results if "alloc" in r.name and r.opportunity_cost > 0)
        alloc_total = sum(1 for r in results if "alloc" in r.name)

        avg_opp_cost = float(np.mean([r.opportunity_cost for r in results]))

        rules = []
        if timing_total > 0 and timing_better / timing_total > 0.6:
            rules.append("Timing decisions frequently suboptimal — consider systematic entry/exit rules")
        if alloc_total > 0 and alloc_better / alloc_total > 0.6:
            rules.append("Allocation decisions frequently suboptimal — consider model-based allocation")

        for r in by_opp_cost[:3]:
            if r.opportunity_cost > 0.01:
                rules.append(f"[{r.name}] {r.insight}")

        return {
            "n_analyses": len(results),
            "avg_opportunity_cost": avg_opp_cost,
            "total_opportunity_cost": float(sum(r.opportunity_cost for r in results)),
            "best_counterfactual": by_opp_cost[0].name if by_opp_cost else None,
            "best_opportunity_cost": float(by_opp_cost[0].opportunity_cost) if by_opp_cost else 0.0,
            "decision_rules": rules,
            "confidence_weighted_insight": float(
                sum(r.opportunity_cost * r.confidence for r in results) /
                max(sum(r.confidence for r in results), 1e-10)
            ),
        }
