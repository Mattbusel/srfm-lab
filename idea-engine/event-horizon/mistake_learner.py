"""
Mistake Learner: extract anti-patterns from losing trades.

What NOT to do is as valuable as what to do. This module:
1. Analyzes every losing trade to find common conditions at entry
2. Builds an "anti-pattern library" of conditions that predict losses
3. Generates "avoidance rules" that veto entries matching anti-patterns
4. Tracks which anti-patterns save the most money when applied
5. Evolves anti-patterns over time as market conditions change

The key insight: a losing trade is information. The conditions that were
present at entry but NOT present in winning trades are the anti-signal.
"""

from __future__ import annotations
import math
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TradeOutcome:
    """A completed trade with full context at entry."""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    hold_bars: int

    # Context at entry
    regime: str
    bh_mass: float
    volatility: float
    spread_bps: float
    hour_of_day: int
    signal_strength: float
    consciousness_activation: float
    fear_greed_index: float
    fractal_coherence: float
    information_surprise: float
    liquidity_warning: float
    dream_fragility: float

    # Outcome
    is_winner: bool = False
    max_adverse_excursion: float = 0.0  # worst unrealized loss during trade


@dataclass
class AntiPattern:
    """A condition set that predicts losing trades."""
    pattern_id: str
    name: str
    description: str
    conditions: Dict[str, Tuple[str, float]]  # feature -> (operator, threshold)
    n_losses_matched: int = 0
    n_wins_matched: int = 0
    loss_rate: float = 0.0       # fraction of matched trades that lost
    avg_loss_when_matched: float = 0.0
    money_saved_estimate: float = 0.0  # estimated $ saved by avoiding
    confidence: float = 0.0
    created_at: float = 0.0
    is_active: bool = True


class MistakeLearner:
    """
    Learns from losing trades to build avoidance rules.

    Process:
    1. Collect all trade outcomes with entry context
    2. Separate winners and losers
    3. Find features that are significantly different in losers vs winners
    4. Build anti-pattern rules from these features
    5. Score anti-patterns by money saved
    6. Apply as veto filters on future entries
    """

    def __init__(self, min_trades_for_pattern: int = 20):
        self.min_trades = min_trades_for_pattern
        self._trades: List[TradeOutcome] = []
        self._anti_patterns: Dict[str, AntiPattern] = {}
        self._counter = 0
        self._vetoes_issued: int = 0
        self._money_saved: float = 0.0

    def _next_id(self) -> str:
        self._counter += 1
        return f"anti_{self._counter:04d}"

    def record_trade(self, trade: TradeOutcome) -> None:
        """Record a completed trade."""
        trade.is_winner = trade.pnl_pct > 0
        self._trades.append(trade)

        # Re-learn patterns periodically
        if len(self._trades) % 50 == 0:
            self.learn_anti_patterns()

    def learn_anti_patterns(self) -> List[AntiPattern]:
        """Analyze all trades and extract anti-patterns."""
        if len(self._trades) < self.min_trades:
            return []

        winners = [t for t in self._trades if t.is_winner]
        losers = [t for t in self._trades if not t.is_winner]

        if len(losers) < 5 or len(winners) < 5:
            return []

        # Features to compare
        features = [
            ("regime", "categorical"),
            ("bh_mass", "numeric"),
            ("volatility", "numeric"),
            ("spread_bps", "numeric"),
            ("hour_of_day", "numeric"),
            ("signal_strength", "numeric"),
            ("consciousness_activation", "numeric"),
            ("fear_greed_index", "numeric"),
            ("fractal_coherence", "numeric"),
            ("information_surprise", "numeric"),
            ("liquidity_warning", "numeric"),
            ("dream_fragility", "numeric"),
        ]

        new_patterns = []

        for feat_name, feat_type in features:
            if feat_type == "numeric":
                winner_vals = np.array([getattr(t, feat_name, 0) for t in winners])
                loser_vals = np.array([getattr(t, feat_name, 0) for t in losers])

                if winner_vals.std() < 1e-10 or loser_vals.std() < 1e-10:
                    continue

                # Is this feature significantly different in losers?
                w_mean = float(winner_vals.mean())
                l_mean = float(loser_vals.mean())
                pooled_std = float(np.sqrt((winner_vals.var() + loser_vals.var()) / 2))

                if pooled_std < 1e-10:
                    continue

                effect_size = abs(l_mean - w_mean) / pooled_std

                if effect_size > 0.5:  # medium+ effect size
                    # Create anti-pattern
                    if l_mean > w_mean:
                        # Losers have HIGHER values -> avoid high values
                        threshold = w_mean + 0.5 * pooled_std
                        op = ">"
                        desc = f"Avoid when {feat_name} > {threshold:.3f} (losers avg {l_mean:.3f} vs winners {w_mean:.3f})"
                    else:
                        # Losers have LOWER values -> avoid low values
                        threshold = w_mean - 0.5 * pooled_std
                        op = "<"
                        desc = f"Avoid when {feat_name} < {threshold:.3f} (losers avg {l_mean:.3f} vs winners {w_mean:.3f})"

                    pattern = AntiPattern(
                        pattern_id=self._next_id(),
                        name=f"avoid_{feat_name}_{op}",
                        description=desc,
                        conditions={feat_name: (op, threshold)},
                        confidence=min(1.0, effect_size / 2),
                        created_at=time.time(),
                    )

                    # Score against historical trades
                    pattern = self._score_pattern(pattern)
                    if pattern.loss_rate > 0.6 and pattern.n_losses_matched >= 5:
                        new_patterns.append(pattern)
                        self._anti_patterns[pattern.pattern_id] = pattern

            elif feat_type == "categorical":
                # Regime-based anti-patterns
                loser_regimes = Counter(getattr(t, feat_name, "") for t in losers)
                winner_regimes = Counter(getattr(t, feat_name, "") for t in winners)

                for regime, l_count in loser_regimes.items():
                    w_count = winner_regimes.get(regime, 0)
                    total = l_count + w_count
                    if total >= 10:
                        loss_rate = l_count / total
                        if loss_rate > 0.65:
                            pattern = AntiPattern(
                                pattern_id=self._next_id(),
                                name=f"avoid_{feat_name}_{regime}",
                                description=f"Avoid trading in {regime} regime ({loss_rate:.0%} loss rate, {total} trades)",
                                conditions={feat_name: ("==", regime)},
                                n_losses_matched=l_count,
                                n_wins_matched=w_count,
                                loss_rate=loss_rate,
                                confidence=min(1.0, total / 50),
                                created_at=time.time(),
                            )
                            new_patterns.append(pattern)
                            self._anti_patterns[pattern.pattern_id] = pattern

        return new_patterns

    def _score_pattern(self, pattern: AntiPattern) -> AntiPattern:
        """Score an anti-pattern against all historical trades."""
        matched_losses = 0
        matched_wins = 0
        loss_amounts = []

        for trade in self._trades:
            if self._matches(trade, pattern):
                if trade.is_winner:
                    matched_wins += 1
                else:
                    matched_losses += 1
                    loss_amounts.append(abs(trade.pnl_pct))

        total = matched_losses + matched_wins
        pattern.n_losses_matched = matched_losses
        pattern.n_wins_matched = matched_wins
        pattern.loss_rate = matched_losses / max(total, 1)
        pattern.avg_loss_when_matched = float(np.mean(loss_amounts)) if loss_amounts else 0
        pattern.money_saved_estimate = pattern.avg_loss_when_matched * matched_losses

        return pattern

    def _matches(self, trade: TradeOutcome, pattern: AntiPattern) -> bool:
        """Check if a trade matches an anti-pattern's conditions."""
        for feat, (op, threshold) in pattern.conditions.items():
            value = getattr(trade, feat, None)
            if value is None:
                return False
            if op == ">" and not (value > threshold):
                return False
            if op == "<" and not (value < threshold):
                return False
            if op == ">=" and not (value >= threshold):
                return False
            if op == "==" and value != threshold:
                return False
        return True

    def should_veto(self, entry_context: Dict[str, float]) -> Optional[AntiPattern]:
        """
        Check if the current entry conditions match any active anti-pattern.
        Returns the matching pattern if veto is recommended, else None.
        """
        for pattern in self._anti_patterns.values():
            if not pattern.is_active or pattern.confidence < 0.3:
                continue

            all_match = True
            for feat, (op, threshold) in pattern.conditions.items():
                value = entry_context.get(feat)
                if value is None:
                    all_match = False
                    break
                if op == ">" and not (value > threshold):
                    all_match = False
                    break
                if op == "<" and not (value < threshold):
                    all_match = False
                    break
                if op == "==" and value != threshold:
                    all_match = False
                    break

            if all_match:
                self._vetoes_issued += 1
                self._money_saved += pattern.avg_loss_when_matched
                return pattern

        return None

    def get_top_anti_patterns(self, n: int = 10) -> List[Dict]:
        """Get the most valuable anti-patterns."""
        patterns = sorted(self._anti_patterns.values(),
                           key=lambda p: p.money_saved_estimate, reverse=True)
        return [
            {
                "name": p.name,
                "description": p.description,
                "loss_rate": p.loss_rate,
                "n_matched": p.n_losses_matched + p.n_wins_matched,
                "money_saved_est": p.money_saved_estimate,
                "confidence": p.confidence,
                "active": p.is_active,
            }
            for p in patterns[:n]
        ]

    def get_stats(self) -> Dict:
        return {
            "total_trades_analyzed": len(self._trades),
            "active_anti_patterns": sum(1 for p in self._anti_patterns.values() if p.is_active),
            "total_vetoes_issued": self._vetoes_issued,
            "estimated_money_saved": self._money_saved,
            "win_rate_all": sum(1 for t in self._trades if t.is_winner) / max(len(self._trades), 1),
        }
