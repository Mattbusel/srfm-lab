"""
Regime-specific trading rules and parameter adaptation.

Defines how strategy parameters should adapt in each detected regime.
Used to generate regime-specific parameter recommendations for IAE.

Implements:
- RegimeTradingRules: parameter lookup and application per regime
- RuleBacktester: test regime-adaptive rules vs static rules historically
- RuleOptimizer: optuna-based search for optimal per-regime multipliers
- RuleConsistencyChecker: verify no parameter cliffs at regime boundaries
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# BH constants
# ---------------------------------------------------------------------------

BH_MASS_THRESH = 1.92
BH_DECAY = 0.924
BH_COLLAPSE = 0.992
HURST_TREND = 0.58
HURST_MR = 0.42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    """Single trade in a regime backtest."""
    entry_bar: int
    exit_bar: int
    regime: str
    entry_price: float
    exit_price: float
    position_scale: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    """Summary of a regime-adaptive backtest."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    avg_hold_bars: float
    trades: List[BacktestTrade]
    equity_curve: np.ndarray
    regime_pnl: Dict[str, float]    # PnL breakdown by regime


@dataclass
class OptimizationResult:
    """Result of regime multiplier optimization."""
    regime: str
    best_params: Dict[str, float]
    best_sharpe: float
    n_trials: int
    param_history: List[Dict[str, Any]]


@dataclass
class ConsistencyCheck:
    """Result of a consistency check between two adjacent regimes."""
    regime_a: str
    regime_b: str
    param_name: str
    value_a: float
    value_b: float
    ratio: float                 # value_b / value_a
    cliff_detected: bool         # ratio > cliff_threshold
    cliff_threshold: float


@dataclass
class ConsistencyReport:
    """Report of all consistency checks across regime boundaries."""
    checks: List[ConsistencyCheck]
    n_cliffs: int
    cliff_pairs: List[Tuple[str, str, str]]   # (regime_a, regime_b, param)
    passed: bool


# ---------------------------------------------------------------------------
# Regime trading rules definition
# ---------------------------------------------------------------------------

class RegimeTradingRules:
    """
    Defines how strategy parameters should adapt in each detected regime.
    Used to generate regime-specific parameter recommendations for IAE.

    Each regime entry contains multipliers that scale a base parameter set.
    The semantics: adapted_param = base_param * multiplier.

    Parameters
    ----------
    custom_rules : optional dict to override defaults
    """

    # Default rules -- one entry per regime name
    # position_scale    : fraction of max position size to use
    # stop_loss_atr_mult: ATR multiplier for stop loss distance
    # entry_zscore      : required z-score threshold before entry
    # hold_bars_min     : minimum bars to hold before exit allowed
    # tp_atr_mult       : ATR multiplier for take-profit
    # vol_target_scale  : multiplier on volatility targeting
    RULES: Dict[str, Dict[str, float]] = {
        "BULL_TREND": {
            "position_scale": 1.0,
            "stop_loss_atr_mult": 2.0,
            "entry_zscore": 1.5,
            "hold_bars_min": 3,
            "tp_atr_mult": 4.0,
            "vol_target_scale": 1.0,
        },
        "BEAR_TREND": {
            "position_scale": 0.5,
            "stop_loss_atr_mult": 1.5,
            "entry_zscore": 2.0,
            "hold_bars_min": 2,
            "tp_atr_mult": 3.0,
            "vol_target_scale": 0.75,
        },
        "HIGH_VOL": {
            "position_scale": 0.3,
            "stop_loss_atr_mult": 3.0,
            "entry_zscore": 2.5,
            "hold_bars_min": 1,
            "tp_atr_mult": 5.0,
            "vol_target_scale": 0.5,
        },
        "RANGING": {
            "position_scale": 0.7,
            "stop_loss_atr_mult": 1.8,
            "entry_zscore": 1.2,
            "hold_bars_min": 2,
            "tp_atr_mult": 2.5,
            "vol_target_scale": 0.9,
        },
        "CRISIS": {
            "position_scale": 0.1,
            "stop_loss_atr_mult": 4.0,
            "entry_zscore": 3.5,
            "hold_bars_min": 1,
            "tp_atr_mult": 6.0,
            "vol_target_scale": 0.25,
        },
        "RECOVERY": {
            "position_scale": 0.6,
            "stop_loss_atr_mult": 2.5,
            "entry_zscore": 1.8,
            "hold_bars_min": 3,
            "tp_atr_mult": 4.5,
            "vol_target_scale": 0.85,
        },
    }

    # BH-physics-derived fallback rules for unknown regimes
    # Use BH_DECAY as a conservative scale factor
    _FALLBACK: Dict[str, float] = {
        "position_scale": BH_DECAY,
        "stop_loss_atr_mult": 2.0,
        "entry_zscore": 2.0,
        "hold_bars_min": 2,
        "tp_atr_mult": 3.5,
        "vol_target_scale": BH_DECAY,
    }

    def __init__(self, custom_rules: Optional[Dict[str, Dict[str, float]]] = None):
        self.rules = {k: dict(v) for k, v in self.RULES.items()}
        if custom_rules:
            for regime, params in custom_rules.items():
                if regime in self.rules:
                    self.rules[regime].update(params)
                else:
                    self.rules[regime] = dict(params)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def get_rules(self, regime: str) -> Dict[str, float]:
        """
        Return the parameter rules for the given regime.

        Falls back to BH-decay-scaled defaults for unknown regimes.

        Parameters
        ----------
        regime : regime label string

        Returns
        -------
        dict of param_name -> value
        """
        if regime in self.rules:
            return dict(self.rules[regime])
        warnings.warn(f"Unknown regime '{regime}' -- using fallback rules.")
        return dict(self._FALLBACK)

    def apply_rules(self, base_params: Dict[str, Any], regime: str) -> Dict[str, Any]:
        """
        Scale base strategy parameters by the regime-specific multipliers.

        For each key in base_params, if it also appears in regime rules,
        the rule value replaces the base value for scalar floats.
        For keys not in regime rules, the base value is kept unchanged.

        Parameters
        ----------
        base_params : dict of param_name -> base value
        regime      : regime label

        Returns
        -------
        dict of param_name -> adapted value
        """
        regime_rules = self.get_rules(regime)
        adapted = dict(base_params)
        for key, rule_val in regime_rules.items():
            if key in adapted:
                base_val = adapted[key]
                if isinstance(base_val, (int, float)):
                    # Multiplicative scaling for most params
                    # Exception: entry_zscore and hold_bars_min are absolute values
                    if key in ("entry_zscore", "hold_bars_min"):
                        adapted[key] = rule_val
                    elif key == "position_scale":
                        adapted[key] = float(base_val) * rule_val
                    else:
                        adapted[key] = float(base_val) * rule_val
            else:
                # Add rule param if not in base
                adapted[key] = rule_val
        return adapted

    def all_regimes(self) -> List[str]:
        """Return sorted list of all known regime names."""
        return sorted(self.rules.keys())

    def rules_dataframe(self) -> pd.DataFrame:
        """Return rules as a tidy DataFrame with regimes as rows."""
        return pd.DataFrame(self.rules).T


# ---------------------------------------------------------------------------
# RuleBacktester
# ---------------------------------------------------------------------------

class RuleBacktester:
    """
    Test regime-adaptive rules versus static rules on historical data.

    Uses a simplified bar-by-bar simulation:
    - Generates entry signals based on z-score of price vs rolling mean
    - Applies regime-specific position_scale, stop_loss_atr_mult, hold_bars_min
    - Compares adaptive strategy against static baseline

    Parameters
    ----------
    trading_rules  : RegimeTradingRules instance
    static_params  : base params for the static (non-adaptive) strategy
    """

    def __init__(
        self,
        trading_rules: RegimeTradingRules,
        static_params: Optional[Dict[str, float]] = None,
    ):
        self.trading_rules = trading_rules
        self.static_params = static_params or {
            "position_scale": 1.0,
            "stop_loss_atr_mult": 2.0,
            "entry_zscore": 1.5,
            "hold_bars_min": 2,
            "tp_atr_mult": 3.5,
        }

    def run(
        self,
        prices: np.ndarray,
        regime_labels: List[str],
        zscore_window: int = 20,
        atr_window: int = 14,
        annual_bars: int = 252,
    ) -> Tuple[BacktestResult, BacktestResult]:
        """
        Run adaptive vs static backtest on price series with regime labels.

        Parameters
        ----------
        prices        : (n,) price array
        regime_labels : (n,) list of regime strings, aligned with prices
        zscore_window : lookback for entry z-score (default 20 bars)
        atr_window    : ATR lookback (default 14 bars)
        annual_bars   : bars per year for Sharpe annualization

        Returns
        -------
        (adaptive_result, static_result) -- BacktestResult pair
        """
        n = len(prices)
        assert len(regime_labels) == n, "prices and regime_labels must have same length"

        # Compute rolling ATR (simplified: std of returns * price)
        log_rets = np.diff(np.log(prices + 1e-10), prepend=np.log(prices[0] + 1e-10))
        atr = self._rolling_std(log_rets, atr_window) * prices

        # Compute z-score signal
        roll_mean = self._rolling_mean(prices, zscore_window)
        roll_std = self._rolling_std(prices, zscore_window)
        zscore = (prices - roll_mean) / (roll_std + 1e-10)

        adaptive_trades = self._simulate(
            prices, regime_labels, zscore, atr, adaptive=True
        )
        static_trades = self._simulate(
            prices, regime_labels, zscore, atr, adaptive=False
        )

        adaptive_res = self._compute_metrics(
            "adaptive", adaptive_trades, prices, annual_bars, regime_labels
        )
        static_res = self._compute_metrics(
            "static", static_trades, prices, annual_bars, regime_labels
        )
        return adaptive_res, static_res

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        prices: np.ndarray,
        regime_labels: List[str],
        zscore: np.ndarray,
        atr: np.ndarray,
        adaptive: bool,
    ) -> List[BacktestTrade]:
        n = len(prices)
        trades: List[BacktestTrade] = []

        in_trade = False
        entry_bar = 0
        entry_price = 0.0
        stop_price = 0.0
        tp_price = 0.0
        pos_scale = 1.0
        hold_min = 2
        regime_at_entry = "UNKNOWN"

        for t in range(20, n):
            regime = regime_labels[t]

            if adaptive:
                rules = self.trading_rules.get_rules(regime)
            else:
                rules = self.static_params

            entry_z = float(rules.get("entry_zscore", 1.5))
            sl_mult = float(rules.get("stop_loss_atr_mult", 2.0))
            tp_mult = float(rules.get("tp_atr_mult", 3.5))
            hold_min = int(rules.get("hold_bars_min", 2))
            pos_scale = float(rules.get("position_scale", 1.0))

            if in_trade:
                bars_held = t - entry_bar
                # Stop loss
                if prices[t] <= stop_price or prices[t] >= tp_price:
                    exit_p = stop_price if prices[t] <= stop_price else tp_price
                    pnl_pct = (exit_p - entry_price) / (entry_price + 1e-10)
                    trades.append(BacktestTrade(
                        entry_bar=entry_bar,
                        exit_bar=t,
                        regime=regime_at_entry,
                        entry_price=entry_price,
                        exit_price=exit_p,
                        position_scale=pos_scale,
                        pnl=pnl_pct * pos_scale,
                        pnl_pct=pnl_pct,
                    ))
                    in_trade = False
                # Signal reversal exit after hold_min
                elif bars_held >= hold_min and zscore[t] < 0:
                    pnl_pct = (prices[t] - entry_price) / (entry_price + 1e-10)
                    trades.append(BacktestTrade(
                        entry_bar=entry_bar,
                        exit_bar=t,
                        regime=regime_at_entry,
                        entry_price=entry_price,
                        exit_price=prices[t],
                        position_scale=pos_scale,
                        pnl=pnl_pct * pos_scale,
                        pnl_pct=pnl_pct,
                    ))
                    in_trade = False
            else:
                # Entry: z-score exceeds threshold (long only for simplicity)
                if zscore[t] > entry_z and pos_scale > 0:
                    in_trade = True
                    entry_bar = t
                    entry_price = prices[t]
                    stop_price = prices[t] - sl_mult * atr[t]
                    tp_price = prices[t] + tp_mult * atr[t]
                    regime_at_entry = regime

        # Close any open trade at end
        if in_trade and n > entry_bar:
            pnl_pct = (prices[-1] - entry_price) / (entry_price + 1e-10)
            trades.append(BacktestTrade(
                entry_bar=entry_bar,
                exit_bar=n - 1,
                regime=regime_at_entry,
                entry_price=entry_price,
                exit_price=prices[-1],
                position_scale=pos_scale,
                pnl=pnl_pct * pos_scale,
                pnl_pct=pnl_pct,
            ))

        return trades

    def _compute_metrics(
        self,
        name: str,
        trades: List[BacktestTrade],
        prices: np.ndarray,
        annual_bars: int,
        regime_labels: List[str],
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(
                strategy_name=name,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                n_trades=0,
                win_rate=0.0,
                avg_hold_bars=0.0,
                trades=[],
                equity_curve=np.ones(len(prices)),
                regime_pnl={},
            )

        pnls = np.array([t.pnl for t in trades])
        wins = np.sum(pnls > 0)
        win_rate = float(wins / len(pnls))
        total_return = float(np.sum(pnls))
        avg_hold = float(np.mean([t.exit_bar - t.entry_bar for t in trades]))

        # Build equity curve
        equity = np.ones(len(prices))
        for tr in trades:
            equity[tr.entry_bar:tr.exit_bar + 1] = (
                equity[tr.entry_bar] * (1.0 + tr.pnl_pct * tr.position_scale)
            )

        # Sharpe from daily PnL
        trade_returns = np.array([t.pnl_pct * t.position_scale for t in trades])
        sr = 0.0
        if len(trade_returns) >= 2 and np.std(trade_returns) > 1e-10:
            sr = (np.mean(trade_returns) / np.std(trade_returns)) * math.sqrt(annual_bars)

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / (running_max + 1e-10)
        max_dd = float(np.min(dd))

        # PnL by regime
        regime_pnl: Dict[str, float] = {}
        for tr in trades:
            regime_pnl[tr.regime] = regime_pnl.get(tr.regime, 0.0) + tr.pnl

        return BacktestResult(
            strategy_name=name,
            total_return=total_return,
            sharpe_ratio=float(sr),
            max_drawdown=max_dd,
            n_trades=len(trades),
            win_rate=win_rate,
            avg_hold_bars=avg_hold,
            trades=trades,
            equity_curve=equity,
            regime_pnl=regime_pnl,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(x, np.nan)
        for i in range(window - 1, len(x)):
            out[i] = np.mean(x[i - window + 1: i + 1])
        return out

    @staticmethod
    def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(x, 1e-10)
        for i in range(window - 1, len(x)):
            out[i] = np.std(x[i - window + 1: i + 1]) + 1e-10
        return out


# ---------------------------------------------------------------------------
# RuleOptimizer
# ---------------------------------------------------------------------------

class RuleOptimizer:
    """
    Optuna-based search for optimal parameter multipliers per regime.

    Searches over the multiplier space for each regime independently,
    using in-sample (IS) data.  Out-of-sample evaluation should be
    performed separately using the returned OptimizationResult.

    Parameters
    ----------
    trading_rules : RegimeTradingRules instance to update with optimal params
    backtester    : RuleBacktester instance
    n_trials      : number of optuna trials per regime (default 50)
    """

    def __init__(
        self,
        trading_rules: RegimeTradingRules,
        backtester: RuleBacktester,
        n_trials: int = 50,
    ):
        self.trading_rules = trading_rules
        self.backtester = backtester
        self.n_trials = n_trials

    def optimize_regime(
        self,
        regime: str,
        prices: np.ndarray,
        regime_labels: List[str],
    ) -> OptimizationResult:
        """
        Search for optimal rules for a single regime using optuna.

        If optuna is not installed, falls back to a grid search over
        a coarse parameter grid.

        Parameters
        ----------
        regime        : regime name to optimize
        prices        : (n,) IS price series
        regime_labels : (n,) labels aligned with prices

        Returns
        -------
        OptimizationResult
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            return self._optuna_search(regime, prices, regime_labels)
        except ImportError:
            warnings.warn("optuna not installed -- falling back to grid search.")
            return self._grid_search(regime, prices, regime_labels)

    def _objective(
        self,
        prices: np.ndarray,
        regime_labels: List[str],
        regime: str,
        params: Dict[str, float],
    ) -> float:
        """Objective: Sharpe ratio of adaptive strategy with given params."""
        custom = {regime: params}
        rules = RegimeTradingRules(custom_rules=custom)
        bt = RuleBacktester(rules, self.backtester.static_params)
        adaptive_res, _ = bt.run(prices, regime_labels)
        return adaptive_res.sharpe_ratio

    def _optuna_search(
        self,
        regime: str,
        prices: np.ndarray,
        regime_labels: List[str],
    ) -> OptimizationResult:
        import optuna

        history: List[Dict[str, Any]] = []
        best_sharpe = -float("inf")
        best_params: Dict[str, float] = {}

        def objective(trial: "optuna.Trial") -> float:
            nonlocal best_sharpe, best_params
            params = {
                "position_scale": trial.suggest_float("position_scale", 0.1, 1.5),
                "stop_loss_atr_mult": trial.suggest_float("stop_loss_atr_mult", 0.5, 5.0),
                "entry_zscore": trial.suggest_float("entry_zscore", 0.5, 4.0),
                "hold_bars_min": trial.suggest_int("hold_bars_min", 1, 10),
                "tp_atr_mult": trial.suggest_float("tp_atr_mult", 1.0, 8.0),
                "vol_target_scale": trial.suggest_float("vol_target_scale", 0.2, 1.5),
            }
            sharpe = self._objective(prices, regime_labels, regime, params)
            history.append({"params": params, "sharpe": sharpe})
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = dict(params)
            return sharpe

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return OptimizationResult(
            regime=regime,
            best_params=best_params,
            best_sharpe=float(best_sharpe),
            n_trials=self.n_trials,
            param_history=history,
        )

    def _grid_search(
        self,
        regime: str,
        prices: np.ndarray,
        regime_labels: List[str],
    ) -> OptimizationResult:
        """Coarse grid search fallback when optuna is unavailable."""
        pos_scales = [0.3, 0.6, 1.0, 1.3]
        sl_mults = [1.0, 2.0, 3.0]
        entry_zs = [1.0, 1.5, 2.0, 2.5]

        best_sharpe = -float("inf")
        best_params: Dict[str, float] = {}
        history: List[Dict[str, Any]] = []

        for ps in pos_scales:
            for sl in sl_mults:
                for ez in entry_zs:
                    params = {
                        "position_scale": ps,
                        "stop_loss_atr_mult": sl,
                        "entry_zscore": ez,
                        "hold_bars_min": 2,
                        "tp_atr_mult": sl * 2.0,
                        "vol_target_scale": ps,
                    }
                    sharpe = self._objective(prices, regime_labels, regime, params)
                    history.append({"params": params, "sharpe": sharpe})
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = dict(params)

        return OptimizationResult(
            regime=regime,
            best_params=best_params,
            best_sharpe=float(best_sharpe),
            n_trials=len(history),
            param_history=history,
        )


# ---------------------------------------------------------------------------
# RuleConsistencyChecker
# ---------------------------------------------------------------------------

class RuleConsistencyChecker:
    """
    Verify that regime rules don't create parameter cliffs at regime
    boundaries -- large discontinuous jumps that would cause unstable
    behavior near regime transition points.

    A cliff is defined as: |param_a / param_b| > cliff_threshold
    (or its reciprocal > cliff_threshold).

    Parameters
    ----------
    trading_rules    : RegimeTradingRules instance
    cliff_threshold  : ratio above which a cliff is flagged (default 3.0)
    transition_pairs : list of (regime_a, regime_b) pairs to check;
                       if None, all pairs from the transition matrix are used
    """

    def __init__(
        self,
        trading_rules: RegimeTradingRules,
        cliff_threshold: float = 3.0,
        transition_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.trading_rules = trading_rules
        self.cliff_threshold = cliff_threshold
        self.transition_pairs = transition_pairs

    def check(
        self,
        transition_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> ConsistencyReport:
        """
        Check all specified transition pairs for parameter cliffs.

        Parameters
        ----------
        transition_pairs : list of (from_regime, to_regime) tuples;
                           if None, uses self.transition_pairs or all pairs

        Returns
        -------
        ConsistencyReport
        """
        pairs = transition_pairs or self.transition_pairs
        if pairs is None:
            regimes = self.trading_rules.all_regimes()
            pairs = [
                (a, b)
                for i, a in enumerate(regimes)
                for b in regimes[i + 1:]
                if a != b
            ]

        checks: List[ConsistencyCheck] = []
        cliff_pairs: List[Tuple[str, str, str]] = []

        for regime_a, regime_b in pairs:
            rules_a = self.trading_rules.get_rules(regime_a)
            rules_b = self.trading_rules.get_rules(regime_b)

            common_params = set(rules_a.keys()) & set(rules_b.keys())
            for param in sorted(common_params):
                val_a = float(rules_a[param])
                val_b = float(rules_b[param])

                if abs(val_a) < 1e-10 and abs(val_b) < 1e-10:
                    ratio = 1.0
                elif abs(val_a) < 1e-10:
                    ratio = float("inf")
                else:
                    ratio = val_b / val_a

                cliff = ratio > self.cliff_threshold or (
                    ratio > 0 and (1.0 / ratio) > self.cliff_threshold
                )

                cc = ConsistencyCheck(
                    regime_a=regime_a,
                    regime_b=regime_b,
                    param_name=param,
                    value_a=val_a,
                    value_b=val_b,
                    ratio=float(ratio),
                    cliff_detected=cliff,
                    cliff_threshold=self.cliff_threshold,
                )
                checks.append(cc)
                if cliff:
                    cliff_pairs.append((regime_a, regime_b, param))

        return ConsistencyReport(
            checks=checks,
            n_cliffs=len(cliff_pairs),
            cliff_pairs=cliff_pairs,
            passed=len(cliff_pairs) == 0,
        )

    def smooth_cliffs(self, report: ConsistencyReport) -> RegimeTradingRules:
        """
        Return a new RegimeTradingRules with cliffs smoothed via geometric mean.

        For each cliff pair (regime_a, regime_b, param), replace the larger
        value with the geometric mean of the two values.

        Parameters
        ----------
        report : ConsistencyReport from check()

        Returns
        -------
        new RegimeTradingRules with smoothed parameters
        """
        smoothed_overrides: Dict[str, Dict[str, float]] = {}

        for regime_a, regime_b, param in report.cliff_pairs:
            rules_a = self.trading_rules.get_rules(regime_a)
            rules_b = self.trading_rules.get_rules(regime_b)
            val_a = rules_a.get(param, 1.0)
            val_b = rules_b.get(param, 1.0)

            # Geometric mean
            if val_a > 0 and val_b > 0:
                geo_mean = math.sqrt(val_a * val_b)
            else:
                geo_mean = (val_a + val_b) / 2.0

            # Set the larger one to geo_mean (pull down toward center)
            if val_a > val_b:
                smoothed_overrides.setdefault(regime_a, {})[param] = geo_mean
            else:
                smoothed_overrides.setdefault(regime_b, {})[param] = geo_mean

        # Build new rules by merging
        new_rules_dict: Dict[str, Dict[str, float]] = {}
        for regime in self.trading_rules.all_regimes():
            base = dict(self.trading_rules.get_rules(regime))
            overrides = smoothed_overrides.get(regime, {})
            base.update(overrides)
            new_rules_dict[regime] = base

        return RegimeTradingRules(custom_rules=new_rules_dict)

    def report_dataframe(self, report: ConsistencyReport) -> pd.DataFrame:
        """Return consistency checks as a tidy DataFrame."""
        rows = [
            {
                "regime_a": c.regime_a,
                "regime_b": c.regime_b,
                "param": c.param_name,
                "value_a": c.value_a,
                "value_b": c.value_b,
                "ratio": round(c.ratio, 3),
                "cliff": c.cliff_detected,
            }
            for c in report.checks
        ]
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values("cliff", ascending=False)
            .reset_index(drop=True)
        )
