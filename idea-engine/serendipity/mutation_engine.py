"""
mutation_engine.py — Strategy Mutation Engine
==============================================
Applies structured random mutations to the BH trading strategy to produce
novel strategy variants for testing.  Each mutation operator preserves the
overall strategy structure while modifying one aspect at a time — analogous
to genetic mutation in evolutionary algorithms.

8 Mutation Operators
--------------------
1. add_filter        — Add a new market condition filter
2. remove_filter     — Remove an existing filter
3. swap_signal       — Replace one entry signal with another
4. invert_signal     — Trade in the opposite direction on a signal
5. time_shift        — Use a lagged version of a signal
6. frequency_change  — Apply signal at a different bar resolution
7. combine_signals   — AND/OR two signals together
8. scale_mutation    — Multiply a parameter by a random factor (0.5–2.0)

Usage
-----
    mutator  = StrategyMutator(seed=42)
    mutation = mutator.mutate(current_strategy_dict, operator="add_filter")
    print(mutation.description)
    print("Param delta:", mutation.param_delta)

    # or: random mutation
    mutation = mutator.random_mutate(current_strategy_dict)
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Signal / filter catalogues (BH strategy vocabulary)
# ---------------------------------------------------------------------------

ENTRY_SIGNALS: List[str] = [
    "momentum_20",         # 20-bar normalised momentum
    "momentum_50",         # 50-bar normalised momentum
    "ou_zscore",           # Ornstein-Uhlenbeck z-score
    "ema_crossover_9_21",  # 9/21 EMA crossover
    "ema_crossover_21_55", # 21/55 EMA crossover
    "rsi_14",              # RSI(14)
    "rsi_7",               # RSI(7)
    "macd_signal",         # MACD histogram sign change
    "bollinger_squeeze",   # Bollinger Band squeeze breakout
    "vol_breakout",        # Volatility breakout
    "order_flow_imbalance",# Bid/ask volume imbalance
    "funding_rate",        # Perpetual funding rate signal
    "hurst_exponent",      # Hurst exponent trending indicator
    "returns_skew",        # Rolling return skewness
    "entropy_signal",      # Shannon entropy signal
]

FILTERS: List[str] = [
    "regime_hmm",          # Hidden Markov Model regime
    "vol_filter_high",     # Block trading when vol > threshold
    "vol_filter_low",      # Block trading when vol < threshold
    "trend_adx",           # Trade only when ADX > 25
    "time_of_day",         # Restrict to specific session hours
    "spread_filter",       # Block when spread > threshold
    "liquidity_filter",    # Block when order book depth < threshold
    "correlation_filter",  # Block when crypto/BTC correlation breaks down
    "drawdown_gate",       # Reduce size after drawdown
    "news_blackout",       # Pause around scheduled events
    "weekend_filter",      # No positions over weekend
    "funding_filter",      # Filter based on funding rate direction
]

TIMEFRAMES: List[str] = [
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d",
]

TIMEFRAME_MULTIPLIERS: Dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}

LOGICAL_OPERATORS: List[str] = ["AND", "OR", "AND NOT"]

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MutatedStrategy:
    """
    Represents a single strategy variant produced by a mutation operator.

    Attributes
    ----------
    operator : str
        Name of the mutation operator applied.
    description : str
        Human-readable description of the mutation.
    param_delta : dict
        Parameter changes: key → new value or delta.
    rationale : str
        Why this mutation is worth testing.
    estimated_impact : str
        Expected magnitude of change: 'low', 'medium', 'high'.
    components_affected : List[str]
        Which strategy components are touched.
    experiment_json : dict
        Full experiment spec ready for the hypothesis generator.
    """

    operator:             str
    description:          str
    param_delta:          dict
    rationale:            str             = ""
    estimated_impact:     str             = "medium"
    components_affected:  List[str]       = field(default_factory=list)
    experiment_json:      dict            = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"MutatedStrategy(op={self.operator!r}, "
            f"impact={self.estimated_impact!r}, desc={self.description[:60]!r})"
        )


# ---------------------------------------------------------------------------
# StrategyMutator
# ---------------------------------------------------------------------------

class StrategyMutator:
    """
    Applies structured random mutations to strategy parameter dicts.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducible mutations.
    """

    OPERATORS: List[str] = [
        "add_filter",
        "remove_filter",
        "swap_signal",
        "invert_signal",
        "time_shift",
        "frequency_change",
        "combine_signals",
        "scale_mutation",
    ]

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mutate(
        self,
        strategy: dict,
        operator: Optional[str] = None,
    ) -> MutatedStrategy:
        """
        Apply a named (or random) mutation operator to *strategy*.

        Parameters
        ----------
        strategy : dict
            Current strategy parameter dict. Expected keys:
            'signals' (list), 'filters' (list), 'timeframe' (str),
            'params' (dict), 'direction' (str).
        operator : str or None
            One of OPERATORS, or None for random selection.

        Returns
        -------
        MutatedStrategy
        """
        if operator is None:
            operator = self._rng.choice(self.OPERATORS)
        if operator not in self.OPERATORS:
            raise ValueError(
                f"Unknown operator {operator!r}. "
                f"Valid: {self.OPERATORS}"
            )

        dispatch = {
            "add_filter":       self.add_filter,
            "remove_filter":    self.remove_filter,
            "swap_signal":      self.swap_signal,
            "invert_signal":    self.invert_signal,
            "time_shift":       self.time_shift,
            "frequency_change": self.frequency_change,
            "combine_signals":  self.combine_signals,
            "scale_mutation":   self.scale_mutation,
        }
        return dispatch[operator](strategy)

    def random_mutate(self, strategy: dict) -> MutatedStrategy:
        """Apply a uniformly random mutation operator."""
        return self.mutate(strategy, operator=None)

    def batch_mutate(
        self,
        strategy: dict,
        n: int = 5,
        allow_duplicates: bool = False,
    ) -> List[MutatedStrategy]:
        """
        Generate *n* mutations, each using a different operator.

        Parameters
        ----------
        strategy : dict
        n : int
        allow_duplicates : bool
            If False, each operator is used at most once.

        Returns
        -------
        List[MutatedStrategy]
        """
        if allow_duplicates:
            ops = [self._rng.choice(self.OPERATORS) for _ in range(n)]
        else:
            ops = self._rng.sample(self.OPERATORS, min(n, len(self.OPERATORS)))

        return [self.mutate(strategy, op) for op in ops]

    # ------------------------------------------------------------------
    # Operator implementations
    # ------------------------------------------------------------------

    def add_filter(self, strategy: dict) -> MutatedStrategy:
        """
        Operator 1: Add a new market condition filter.

        Picks a filter not already in the strategy's active filter list
        and adds it as a new trading gate.

        Parameters
        ----------
        strategy : dict

        Returns
        -------
        MutatedStrategy
        """
        active_filters: List[str] = strategy.get("filters", [])
        available = [f for f in FILTERS if f not in active_filters]
        if not available:
            available = FILTERS

        new_filter = self._rng.choice(available)
        threshold   = self._rng.uniform(0.2, 0.8)

        description = (
            f"Add filter '{new_filter}' as a mandatory trading gate "
            f"(threshold={threshold:.2f})."
        )
        param_delta = {
            "add_filter":           new_filter,
            "filter_threshold":     round(threshold, 4),
            "filter_logic":         "AND",
        }
        rationale = (
            f"Adding '{new_filter}' constrains the signal space to regimes "
            f"where the strategy has historically had more edge. "
            f"Expected to reduce trade count by 20–40% while improving quality."
        )
        return MutatedStrategy(
            operator             = "add_filter",
            description          = description,
            param_delta          = param_delta,
            rationale            = rationale,
            estimated_impact     = "medium",
            components_affected  = ["regime_filter"],
            experiment_json      = self._build_experiment("add_filter", param_delta, description),
        )

    def remove_filter(self, strategy: dict) -> MutatedStrategy:
        """
        Operator 2: Remove an existing market condition filter.

        Tests whether a current filter is actually improving performance
        or just reducing trade count without quality benefit.

        Parameters
        ----------
        strategy : dict

        Returns
        -------
        MutatedStrategy
        """
        active_filters: List[str] = strategy.get("filters", [])
        if active_filters:
            removed = self._rng.choice(active_filters)
        else:
            removed = self._rng.choice(FILTERS)

        description = (
            f"Remove filter '{removed}' — test whether it actually adds value."
        )
        param_delta = {
            "remove_filter": removed,
        }
        rationale = (
            f"Filter '{removed}' may be over-constraining the signal, "
            f"causing the strategy to miss profitable trades. Removing it "
            f"tests whether the filter is causal or spurious."
        )
        return MutatedStrategy(
            operator             = "remove_filter",
            description          = description,
            param_delta          = param_delta,
            rationale            = rationale,
            estimated_impact     = "medium",
            components_affected  = ["regime_filter"],
            experiment_json      = self._build_experiment("remove_filter", param_delta, description),
        )

    def swap_signal(self, strategy: dict) -> MutatedStrategy:
        """
        Operator 3: Replace one entry signal with another.

        Picks a signal from the active signal list and replaces it with
        a different signal from the catalogue.

        Parameters
        ----------
        strategy : dict

        Returns
        -------
        MutatedStrategy
        """
        active_signals: List[str] = strategy.get("signals", ["momentum_20"])
        old_signal = self._rng.choice(active_signals)

        available_new = [s for s in ENTRY_SIGNALS if s != old_signal]
        new_signal    = self._rng.choice(available_new)

        description = f"Swap signal '{old_signal}' → '{new_signal}'."
        param_delta  = {
            "remove_signal": old_signal,
            "add_signal":    new_signal,
        }
        rationale = (
            f"Signal '{old_signal}' may be losing edge due to crowding or "
            f"regime shift. '{new_signal}' represents an alternative hypothesis "
            f"about the dominant market dynamic."
        )
        return MutatedStrategy(
            operator             = "swap_signal",
            description          = description,
            param_delta          = param_delta,
            rationale            = rationale,
            estimated_impact     = "high",
            components_affected  = ["entry_signal"],
            experiment_json      = self._build_experiment("swap_signal", param_delta, description),
        )

    def invert_signal(self, strategy: dict) -> MutatedStrategy:
        """
        Operator 4: Trade in the opposite direction on a signal.

        Tests whether the primary signal is actually anti-predictive in
        the current regime (contrarian opportunity).

        Parameters
        ----------
        strategy : dict

        Returns
        -------
        MutatedStrategy
        """
        active_signals: List[str] = strategy.get("signals", ["momentum_20"])
        signal = self._rng.choice(active_signals)

        current_direction = strategy.get("direction", "long_short")
        new_direction = "contrarian_" + current_direction

        description = (
            f"Invert signal '{signal}': trade OPPOSITE direction to the "
            f"current signal, testing contrarian behaviour."
        )
        param_delta = {
            "invert_signal":    signal,
            "direction":        new_direction,
        }
        rationale = (
            f"If '{signal}' has been overcrowded or is a lagging indicator, "
            f"inverting it may capture mean-reversion after overextension. "
            f"This tests the contrarian regime hypothesis."
        )
        return MutatedStrategy(
            operator             = "invert_signal",
            description          = description,
            param_delta          = param_delta,
            rationale            = rationale,
            estimated_impact     = "high",
            components_affected  = ["entry_signal", "exit_rule"],
            experiment_json      = self._build_experiment("invert_signal", param_delta, description),
        )

    def time_shift(self, strategy: dict) -> MutatedStrategy:
        """
        Operator 5: Use a lagged version of a signal.

        Adds a lag (1–10 bars) to the primary entry signal to test whether
        delayed confirmation improves precision at the cost of timing.

        Parameters
        ----------
        strategy : dict

        Returns
        -------
        MutatedStrategy
        """
        active_signals: List[str] = strategy.get("signals", ["momentum_20"])
        signal = self._rng.choice(active_signals)

        lag_bars = self._rng.randint(1, 10)

        description = (
            f"Apply a {lag_bars}-bar lag to signal '{signal}' — "
            f"wait for confirmation before entering."
        )
        param_delta = {
            "signal":       signal,
            "lag_bars":     lag_bars,
            "entry_delay":  lag_bars,
        }
        rationale = (
            f"A {lag_bars}-bar lag on '{signal}' delays entry until the "
            f"signal is confirmed, potentially reducing false positives at "
            f"the cost of slightly worse fill prices."
        )
        return MutatedStrategy(
            operator             = "time_shift",
            description          = description,
            param_delta          = param_delta,
            rationale            = rationale,
            estimated_impact     = "medium",
            components_affected  = ["entry_signal"],
            experiment_json      = self._build_experiment("time_shift", param_delta, description),
        )

    def frequency_change(self, strategy: dict) -> MutatedStrategy:
        """
        Operator 6: Apply the signal at a different bar resolution.

        Tests the signal on a higher or lower timeframe to find the
        timeframe where it has the most predictive power.

        Parameters
        ----------
        strategy : dict

        Returns
        -------
        MutatedStrategy
        """
        current_tf = strategy.get("timeframe", "1h")
        current_idx = TIMEFRAMES.index(current_tf) if current_tf in TIMEFRAMES else 5

        # Move 1–3 steps up or down the timeframe ladder
        delta = self._rng.choice([-2, -1, 1, 2, 3])
        new_idx = max(0, min(len(TIMEFRAMES) - 1, current_idx + delta))
        new_tf  = TIMEFRAMES[new_idx]

        direction = "higher" if new_idx > current_idx else "lower"
        description = (
            f"Change primary timeframe from '{current_tf}' to '{new_tf}' "
            f"({direction} resolution)."
        )
        param_delta = {
            "old_timeframe": current_tf,
            "new_timeframe": new_tf,
        }
        impact = "high" if abs(new_idx - current_idx) >= 2 else "medium"
        rationale = (
            f"Testing '{new_tf}' explores whether the signal's edge exists at "
            f"a {direction} frequency. {direction.capitalize()}-frequency signals "
            f"have {'less noise but slower reaction' if direction == 'higher' else 'more noise but faster reaction'}."
        )
        return MutatedStrategy(
            operator             = "frequency_change",
            description          = description,
            param_delta          = param_delta,
            rationale            = rationale,
            estimated_impact     = impact,
            components_affected  = ["entry_signal", "exit_rule"],
            experiment_json      = self._build_experiment("frequency_change", param_delta, description),
        )

    def combine_signals(self, strategy: dict) -> MutatedStrategy:
        """
        Operator 7: Combine two signals with AND/OR logic.

        Takes two signals from the catalogue and combines them using a
        logical operator to form a composite entry condition.

        Parameters
        ----------
        strategy : dict

        Returns
        -------
        MutatedStrategy
        """
        active: List[str] = strategy.get("signals", ["momentum_20"])

        # Pick one existing and one new signal
        sig_a = self._rng.choice(active)
        remaining = [s for s in ENTRY_SIGNALS if s != sig_a]
        sig_b = self._rng.choice(remaining)
        logic = self._rng.choice(LOGICAL_OPERATORS)

        description = (
            f"Combine signals: ({sig_a}) {logic} ({sig_b}) as composite entry."
        )
        param_delta = {
            "signal_a":      sig_a,
            "signal_b":      sig_b,
            "logic":         logic,
            "composite_mode": "conjunction" if logic == "AND" else "disjunction",
        }
        rationale = (
            f"Requiring both '{sig_a}' {logic} '{sig_b}' to agree before "
            f"entry {'increases precision (fewer, higher-quality trades)' if logic == 'AND' else 'increases recall (more trades, potentially more noise)'}. "
            f"Tests whether signal independence holds in practice."
        )
        impact = "high" if logic == "AND" else "medium"
        return MutatedStrategy(
            operator             = "combine_signals",
            description          = description,
            param_delta          = param_delta,
            rationale            = rationale,
            estimated_impact     = impact,
            components_affected  = ["entry_signal"],
            experiment_json      = self._build_experiment("combine_signals", param_delta, description),
        )

    def scale_mutation(self, strategy: dict) -> MutatedStrategy:
        """
        Operator 8: Multiply a parameter by a random factor (0.5–2.0).

        Scales a numeric parameter to explore the neighbourhood around
        the current value — a local parameter sweep.

        Parameters
        ----------
        strategy : dict

        Returns
        -------
        MutatedStrategy
        """
        params: dict = strategy.get("params", {})

        # Default scalable parameters if strategy has none
        scalable_params = {
            "entry_threshold":    params.get("entry_threshold", 1.5),
            "lookback":           params.get("lookback", 20),
            "stop_loss_atr":      params.get("stop_loss_atr", 2.0),
            "kelly_fraction":     params.get("kelly_fraction", 0.25),
            "regime_lookback":    params.get("regime_lookback", 50),
            "vol_lookback":       params.get("vol_lookback", 30),
        }

        param_key = self._rng.choice(list(scalable_params.keys()))
        old_value = scalable_params[param_key]

        # Scale factor: log-uniform between 0.5 and 2.0
        import math
        log_scale = self._rng.uniform(math.log(0.5), math.log(2.0))
        scale     = math.exp(log_scale)
        new_value = round(old_value * scale, 4)

        description = (
            f"Scale parameter '{param_key}': "
            f"{old_value} → {new_value} (×{scale:.2f})."
        )
        param_delta = {
            param_key:    new_value,
            "scale_factor": round(scale, 4),
            "old_value":    old_value,
        }
        direction = "increase" if scale > 1.0 else "decrease"
        rationale = (
            f"A {scale:.1f}× {direction} in '{param_key}' explores the local "
            f"parameter landscape. If performance improves, the current value "
            f"may not be at the optimum."
        )
        impact = "low" if 0.8 < scale < 1.25 else "medium"
        return MutatedStrategy(
            operator             = "scale_mutation",
            description          = description,
            param_delta          = param_delta,
            rationale            = rationale,
            estimated_impact     = impact,
            components_affected  = ["entry_signal", "position_sizing"],
            experiment_json      = self._build_experiment("scale_mutation", param_delta, description),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_experiment(operator: str, param_delta: dict, description: str) -> dict:
        """Build a standardised experiment spec dict."""
        return {
            "operator":        operator,
            "description":     description,
            "param_delta":     param_delta,
            "experiment_type": "mutation",
            "source":          "mutation_engine",
        }

    def apply_param_delta(self, strategy: dict, mutation: MutatedStrategy) -> dict:
        """
        Apply a MutatedStrategy's param_delta to a strategy dict and
        return the new strategy dict (does not mutate the original).

        Parameters
        ----------
        strategy : dict
        mutation : MutatedStrategy

        Returns
        -------
        dict
        """
        import copy
        new_strategy = copy.deepcopy(strategy)
        delta = mutation.param_delta

        # Handle structured operators
        if "add_filter" in delta:
            new_strategy.setdefault("filters", [])
            if delta["add_filter"] not in new_strategy["filters"]:
                new_strategy["filters"].append(delta["add_filter"])

        if "remove_filter" in delta:
            filters = new_strategy.get("filters", [])
            new_strategy["filters"] = [f for f in filters if f != delta["remove_filter"]]

        if "remove_signal" in delta:
            signals = new_strategy.get("signals", [])
            new_strategy["signals"] = [s for s in signals if s != delta["remove_signal"]]
        if "add_signal" in delta:
            new_strategy.setdefault("signals", [])
            if delta["add_signal"] not in new_strategy["signals"]:
                new_strategy["signals"].append(delta["add_signal"])

        if "new_timeframe" in delta:
            new_strategy["timeframe"] = delta["new_timeframe"]

        # Numeric param updates
        for k, v in delta.items():
            if k not in ("add_filter", "remove_filter", "remove_signal",
                         "add_signal", "new_timeframe", "old_timeframe",
                         "scale_factor", "old_value"):
                new_strategy.setdefault("params", {})[k] = v

        return new_strategy

    def __repr__(self) -> str:
        return f"StrategyMutator(operators={self.OPERATORS})"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    sample_strategy = {
        "signals":   ["momentum_20", "ou_zscore"],
        "filters":   ["regime_hmm", "vol_filter_high"],
        "timeframe": "1h",
        "direction": "long_short",
        "params": {
            "entry_threshold": 1.5,
            "lookback":        20,
            "stop_loss_atr":   2.0,
            "kelly_fraction":  0.25,
        },
    }

    mutator = StrategyMutator(seed=42)
    print("Running all 8 mutation operators:\n")
    for op in StrategyMutator.OPERATORS:
        mutation = mutator.mutate(sample_strategy, operator=op)
        print(f"[{op:20s}] {mutation.description[:70]}")
        print(f"  impact={mutation.estimated_impact}  params={mutation.param_delta}")
        print()
