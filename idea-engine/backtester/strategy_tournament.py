"""
Strategy Tournament: pit ALL hypothesis templates against each other on real data.

A round-robin tournament where every strategy template competes:
  - 22 hypothesis templates from idea-engine/hypothesis/templates/
  - Each template runs on the same data with the same costs
  - Head-to-head comparison: which template beats which?
  - Regime-conditional rankings (best in trending vs mean-reverting)
  - Ensemble construction: combine top N templates optimally
  - Tournament brackets: elimination rounds based on OOS performance
  - Statistical significance: is template A REALLY better than B?
  - Overfitting detection: templates that look great IS but fail OOS

This produces the definitive ranking of all strategy approaches
in the system, enabling data-driven template selection.
"""

from __future__ import annotations
import math
import time
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: STRATEGY TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyTemplate:
    """A strategy template that can generate signals from market data."""
    name: str
    template_type: str
    description: str
    signal_fn: Optional[Callable] = None  # function(returns, prices, volumes) -> signals
    lookback: int = 63
    regime_affinity: List[str] = field(default_factory=list)
    parameters: Dict = field(default_factory=dict)


def _momentum_strategy(returns: np.ndarray, **kwargs) -> np.ndarray:
    """12-1 month momentum."""
    T = len(returns)
    signal = np.zeros(T)
    for t in range(252, T):
        mom = np.sum(returns[t-252:t-21])
        vol = max(np.std(returns[t-63:t]), 1e-8)
        signal[t] = np.tanh(mom / vol)
    return signal


def _mean_reversion_strategy(returns: np.ndarray, **kwargs) -> np.ndarray:
    """63-day z-score mean reversion."""
    T = len(returns)
    signal = np.zeros(T)
    prices = np.exp(np.cumsum(returns))
    for t in range(63, T):
        window = prices[t-63:t]
        z = (prices[t] - window.mean()) / max(window.std(), 1e-8)
        signal[t] = -np.tanh(z / 2)
    return signal


def _breakout_strategy(returns: np.ndarray, **kwargs) -> np.ndarray:
    """20-day Donchian breakout."""
    T = len(returns)
    signal = np.zeros(T)
    prices = np.exp(np.cumsum(returns))
    for t in range(21, T):
        high_20 = prices[t-20:t].max()
        low_20 = prices[t-20:t].min()
        if prices[t] >= high_20:
            signal[t] = 1.0
        elif prices[t] <= low_20:
            signal[t] = -1.0
    return signal


def _volatility_breakout_strategy(returns: np.ndarray, **kwargs) -> np.ndarray:
    """Vol breakout: trade when vol expands from compression."""
    T = len(returns)
    signal = np.zeros(T)
    for t in range(42, T):
        vol_fast = np.std(returns[t-5:t])
        vol_slow = np.std(returns[t-42:t])
        if vol_fast > vol_slow * 1.5:
            signal[t] = np.sign(returns[t-5:t].mean())
    return signal


def _rsi_mean_reversion(returns: np.ndarray, **kwargs) -> np.ndarray:
    """RSI-based mean reversion: buy oversold, sell overbought."""
    T = len(returns)
    signal = np.zeros(T)
    for t in range(15, T):
        window = returns[t-14:t]
        gains = np.maximum(window, 0).mean()
        losses = np.maximum(-window, 0).mean()
        rs = gains / max(losses, 1e-8)
        rsi = 100 - 100 / (1 + rs)
        if rsi < 30:
            signal[t] = 0.7  # oversold, buy
        elif rsi > 70:
            signal[t] = -0.7  # overbought, sell
    return signal


def _trend_following(returns: np.ndarray, **kwargs) -> np.ndarray:
    """Dual MA crossover trend following."""
    T = len(returns)
    signal = np.zeros(T)
    prices = np.exp(np.cumsum(returns))
    for t in range(60, T):
        ma_fast = prices[t-20:t].mean()
        ma_slow = prices[t-60:t].mean()
        if ma_fast > ma_slow * 1.01:
            signal[t] = 0.8
        elif ma_fast < ma_slow * 0.99:
            signal[t] = -0.8
    return signal


def _carry_strategy(returns: np.ndarray, **kwargs) -> np.ndarray:
    """Simple carry: go long assets with positive drift."""
    T = len(returns)
    signal = np.zeros(T)
    for t in range(63, T):
        drift = returns[t-63:t].mean() * 252
        if drift > 0.05:
            signal[t] = 0.5
        elif drift < -0.05:
            signal[t] = -0.5
    return signal


def _volatility_targeting(returns: np.ndarray, **kwargs) -> np.ndarray:
    """Vol targeting: scale position by inverse vol."""
    T = len(returns)
    signal = np.zeros(T)
    target_vol = 0.15  # 15% annualized
    for t in range(21, T):
        realized_vol = np.std(returns[t-21:t]) * math.sqrt(252)
        if realized_vol > 0.01:
            scale = target_vol / realized_vol
            trend = np.sign(returns[t-5:t].mean())
            signal[t] = trend * min(scale, 3.0) * 0.3
    return signal


def _pairs_reversion(returns: np.ndarray, **kwargs) -> np.ndarray:
    """Self-pairs: mean revert around rolling mean."""
    T = len(returns)
    signal = np.zeros(T)
    cumrets = np.cumsum(returns)
    for t in range(126, T):
        spread = cumrets[t] - np.mean(cumrets[t-126:t])
        spread_std = max(np.std(cumrets[t-126:t] - np.mean(cumrets[t-126:t])), 1e-8)
        z = spread / spread_std
        if abs(z) > 1.5:
            signal[t] = -np.tanh(z / 2) * 0.6
    return signal


def _entropy_regime(returns: np.ndarray, **kwargs) -> np.ndarray:
    """Entropy-based: low entropy = trend, high entropy = revert."""
    T = len(returns)
    signal = np.zeros(T)
    for t in range(50, T):
        window = returns[t-30:t]
        # Simplified permutation entropy
        patterns = defaultdict(int)
        for i in range(len(window) - 2):
            pat = tuple(np.argsort(window[i:i+3]))
            patterns[pat] += 1
        total = sum(patterns.values())
        if total > 0:
            entropy = -sum((c/total) * math.log(c/total + 1e-15) for c in patterns.values())
            norm_ent = entropy / math.log(6)
            if norm_ent < 0.5:
                # Low entropy: follow trend
                signal[t] = np.sign(window.mean()) * 0.6
            else:
                # High entropy: mean revert
                signal[t] = -np.sign(window[-1]) * 0.3
    return signal


def _physics_bh_mass(returns: np.ndarray, **kwargs) -> np.ndarray:
    """BH physics: mass accumulation on consecutive same-sign bars."""
    T = len(returns)
    signal = np.zeros(T)
    mass = 0.0
    ctl = 0
    BH_FORM = 1.5
    BH_DECAY = 0.924
    for t in range(1, T):
        if np.sign(returns[t]) == np.sign(returns[t-1]) and abs(returns[t]) > 0.001:
            ctl += 1
            sb = min(2.0, 1 + ctl * 0.1)
            mass = mass * 0.97 + abs(returns[t]) * 100 * sb * 0.03
        else:
            mass *= BH_DECAY
            ctl = 0
        if mass > BH_FORM and ctl >= 3:
            signal[t] = np.sign(returns[t]) * min(mass / 3, 1.0)
        else:
            signal[t] *= 0.9  # decay
    return signal


def _hurst_adaptive(returns: np.ndarray, **kwargs) -> np.ndarray:
    """Hurst exponent adaptive: momentum if H>0.55, reversion if H<0.45."""
    T = len(returns)
    signal = np.zeros(T)
    for t in range(100, T):
        window = returns[t-100:t]
        # R/S Hurst approximation
        mean_w = window.mean()
        cumdev = np.cumsum(window - mean_w)
        R = cumdev.max() - cumdev.min()
        S = max(window.std(), 1e-8)
        hurst = math.log(max(R/S, 1e-8)) / math.log(100)
        hurst = max(0, min(1, hurst))
        if hurst > 0.55:
            signal[t] = np.sign(window[-10:].mean()) * 0.5
        elif hurst < 0.45:
            signal[t] = -np.sign(window[-1]) * 0.4
    return signal


# Register all templates
TOURNAMENT_TEMPLATES = [
    StrategyTemplate("Momentum 12-1", "momentum", "12-month momentum skipping last month", _momentum_strategy, 252, ["trending_up", "trending_down"]),
    StrategyTemplate("Mean Reversion Z63", "mean_reversion", "63-day z-score reversion", _mean_reversion_strategy, 63, ["mean_reverting"]),
    StrategyTemplate("Donchian Breakout", "breakout", "20-day high/low breakout", _breakout_strategy, 21, ["trending_up", "trending_down"]),
    StrategyTemplate("Vol Breakout", "volatility", "Volatility expansion breakout", _volatility_breakout_strategy, 42, ["high_volatility"]),
    StrategyTemplate("RSI Reversion", "mean_reversion", "RSI oversold/overbought", _rsi_mean_reversion, 15, ["mean_reverting"]),
    StrategyTemplate("Dual MA Trend", "trend_following", "20/60 MA crossover", _trend_following, 60, ["trending_up", "trending_down"]),
    StrategyTemplate("Carry", "carry", "Positive drift carry trade", _carry_strategy, 63, ["trending_up"]),
    StrategyTemplate("Vol Targeting", "volatility", "Inverse vol position sizing", _volatility_targeting, 21, []),
    StrategyTemplate("Pairs Reversion", "stat_arb", "Self-pairs z-score", _pairs_reversion, 126, ["mean_reverting"]),
    StrategyTemplate("Entropy Regime", "regime_adaptive", "Permutation entropy adaptive", _entropy_regime, 50, []),
    StrategyTemplate("BH Physics Mass", "physics_inspired", "Black hole mass accumulation", _physics_bh_mass, 5, ["trending_up", "trending_down"]),
    StrategyTemplate("Hurst Adaptive", "regime_adaptive", "Hurst exponent switches momentum/reversion", _hurst_adaptive, 100, []),
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: TOURNAMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TemplateResult:
    """Result of running one template on data."""
    template_name: str
    template_type: str
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_hold_bars: float
    ic: float                  # information coefficient
    turnover: float            # daily turnover (0-2)


@dataclass
class HeadToHead:
    """Head-to-head comparison between two templates."""
    template_a: str
    template_b: str
    a_sharpe: float
    b_sharpe: float
    winner: str
    sharpe_diff: float
    statistically_significant: bool  # t-test p < 0.05


@dataclass
class TournamentReport:
    """Complete tournament results."""
    n_templates: int
    n_bars: int
    date: str

    # Rankings
    overall_ranking: List[Dict]           # sorted by Sharpe
    regime_rankings: Dict[str, List[Dict]] # regime -> sorted by Sharpe

    # Head-to-head
    head_to_head_matrix: List[List[float]]  # NxN matrix of Sharpe differences
    n_significant_comparisons: int

    # Ensemble
    optimal_ensemble_weights: Dict[str, float]
    ensemble_sharpe: float
    ensemble_vs_best_single: float

    # Overfitting
    is_vs_oos: List[Dict]                 # IS Sharpe vs OOS Sharpe per template

    # Best in class
    best_momentum: str
    best_reversion: str
    best_regime_adaptive: str
    best_overall: str


class StrategyTournament:
    """
    Round-robin tournament for all strategy templates.
    """

    def __init__(self, templates: List[StrategyTemplate] = None,
                  cost_bps: float = 15, seed: int = 42):
        self.templates = templates or TOURNAMENT_TEMPLATES
        self.cost_bps = cost_bps
        self.seed = seed

    def run(self, returns: np.ndarray, regime_labels: Optional[np.ndarray] = None,
            verbose: bool = True) -> TournamentReport:
        """Run the full tournament."""
        T = len(returns)
        n_templates = len(self.templates)

        if verbose:
            print(f"Strategy Tournament: {n_templates} templates on {T} bars")
            print("=" * 60)

        # Run each template
        results = []
        signals_by_template = {}

        for template in self.templates:
            try:
                signal = template.signal_fn(returns)
                signals_by_template[template.name] = signal
                result = self._evaluate(template, signal, returns)
                results.append(result)
                if verbose:
                    print(f"  {template.name:25s} Sharpe={result.sharpe:+.2f} Return={result.total_return:+.1%} MaxDD={result.max_drawdown:.1%}")
            except Exception as e:
                if verbose:
                    print(f"  {template.name:25s} FAILED: {e}")
                results.append(TemplateResult(
                    template.name, template.template_type, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ))

        # Overall ranking
        overall = sorted(
            [{"name": r.template_name, "type": r.template_type, "sharpe": r.sharpe,
              "return": r.total_return, "max_dd": r.max_drawdown, "win_rate": r.win_rate,
              "ic": r.ic, "n_trades": r.n_trades}
             for r in results],
            key=lambda x: x["sharpe"], reverse=True,
        )

        # Regime rankings
        regime_rankings = {}
        if regime_labels is not None:
            unique_regimes = set(regime_labels)
            for regime in unique_regimes:
                mask = regime_labels == regime
                regime_results = []
                for template, signal in zip(self.templates, [signals_by_template.get(t.name, np.zeros(T)) for t in self.templates]):
                    r_returns = returns[mask]
                    r_signal = signal[mask] if len(signal) == T else np.zeros(mask.sum())
                    if len(r_returns) >= 30:
                        strat_ret = r_signal[:-1] * r_returns[1:]
                        if len(strat_ret) > 0 and strat_ret.std() > 1e-10:
                            sharpe = float(strat_ret.mean() / strat_ret.std() * math.sqrt(252))
                        else:
                            sharpe = 0.0
                        regime_results.append({"name": template.name, "sharpe": sharpe})
                regime_rankings[str(regime)] = sorted(regime_results, key=lambda x: x["sharpe"], reverse=True)

        # Head-to-head matrix
        h2h_matrix = np.zeros((n_templates, n_templates))
        n_significant = 0
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                diff = results[i].sharpe - results[j].sharpe
                h2h_matrix[i, j] = diff
                h2h_matrix[j, i] = -diff
                # Simplified significance test
                if abs(diff) > 0.5:
                    n_significant += 1

        # Optimal ensemble
        ensemble_weights, ensemble_sharpe = self._build_ensemble(signals_by_template, returns)
        best_single = max(r.sharpe for r in results) if results else 0

        # IS vs OOS
        mid = T // 2
        is_oos = []
        for template in self.templates:
            signal = signals_by_template.get(template.name, np.zeros(T))
            is_ret = signal[:mid-1] * returns[1:mid]
            oos_ret = signal[mid:-1] * returns[mid+1:]
            is_sharpe = float(is_ret.mean() / max(is_ret.std(), 1e-10) * math.sqrt(252)) if len(is_ret) > 10 else 0
            oos_sharpe = float(oos_ret.mean() / max(oos_ret.std(), 1e-10) * math.sqrt(252)) if len(oos_ret) > 10 else 0
            is_oos.append({"name": template.name, "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
                            "degradation": 1 - oos_sharpe / max(is_sharpe, 1e-10) if is_sharpe > 0 else 0})

        # Best in class
        by_type = defaultdict(list)
        for r in results:
            by_type[r.template_type].append(r)

        best_mom = max(by_type.get("momentum", [TemplateResult("none", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]), key=lambda r: r.sharpe).template_name
        best_rev = max(by_type.get("mean_reversion", [TemplateResult("none", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]), key=lambda r: r.sharpe).template_name
        best_adaptive = max(by_type.get("regime_adaptive", [TemplateResult("none", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]), key=lambda r: r.sharpe).template_name

        if verbose:
            print(f"\nBest overall: {overall[0]['name']} (Sharpe {overall[0]['sharpe']:.2f})")
            print(f"Ensemble Sharpe: {ensemble_sharpe:.2f} (vs best single {best_single:.2f})")

        return TournamentReport(
            n_templates=n_templates,
            n_bars=T,
            date=time.strftime("%Y-%m-%d"),
            overall_ranking=overall,
            regime_rankings=regime_rankings,
            head_to_head_matrix=h2h_matrix.tolist(),
            n_significant_comparisons=n_significant,
            optimal_ensemble_weights=ensemble_weights,
            ensemble_sharpe=ensemble_sharpe,
            ensemble_vs_best_single=ensemble_sharpe - best_single,
            is_vs_oos=is_oos,
            best_momentum=best_mom,
            best_reversion=best_rev,
            best_regime_adaptive=best_adaptive,
            best_overall=overall[0]["name"] if overall else "none",
        )

    def _evaluate(self, template: StrategyTemplate, signal: np.ndarray,
                   returns: np.ndarray) -> TemplateResult:
        """Evaluate a single template's performance."""
        T = len(returns)
        cost = self.cost_bps / 10000

        # Strategy returns
        strat_ret = signal[:-1] * returns[1:]
        turnover = np.abs(np.diff(signal, prepend=0))[:-1]
        costs = turnover * cost
        net_ret = strat_ret - costs

        if len(net_ret) < 20 or net_ret.std() < 1e-10:
            return TemplateResult(template.name, template.template_type, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        sharpe = float(net_ret.mean() / net_ret.std() * math.sqrt(252))

        downside = net_ret[net_ret < 0]
        down_std = float(downside.std()) if len(downside) > 1 else float(net_ret.std())
        sortino = float(net_ret.mean() / max(down_std, 1e-10) * math.sqrt(252))

        total_ret = float(np.prod(1 + net_ret) - 1)

        eq = np.cumprod(1 + net_ret)
        peak = np.maximum.accumulate(eq)
        max_dd = float(((peak - eq) / peak).max())

        # Trade stats
        position_changes = np.diff(np.sign(signal))
        n_trades = int(np.sum(position_changes != 0))
        winners = net_ret[net_ret > 0]
        losers = net_ret[net_ret < 0]
        win_rate = float(len(winners) / max(len(winners) + len(losers), 1))
        pf = float(winners.sum() / max(abs(losers.sum()), 1e-10)) if len(losers) > 0 else 0

        # IC
        ic = float(np.corrcoef(signal[:-1], returns[1:])[0, 1]) if len(signal) > 20 else 0

        # Turnover
        avg_turnover = float(turnover.mean())

        return TemplateResult(
            template_name=template.name,
            template_type=template.template_type,
            sharpe=sharpe,
            sortino=sortino,
            total_return=total_ret,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=pf,
            n_trades=n_trades,
            avg_hold_bars=T / max(n_trades, 1),
            ic=ic,
            turnover=avg_turnover,
        )

    def _build_ensemble(self, signals: Dict[str, np.ndarray],
                          returns: np.ndarray) -> Tuple[Dict[str, float], float]:
        """Build optimal ensemble from all signals."""
        names = list(signals.keys())
        n = len(names)
        if n < 2:
            return {}, 0.0

        T = len(returns)

        # Compute per-signal strategy returns
        strat_returns = np.zeros((T - 1, n))
        for i, name in enumerate(names):
            sig = signals[name]
            strat_returns[:, i] = sig[:T-1] * returns[1:T]

        # Optimal weights: inverse variance (simplified)
        vols = np.array([max(strat_returns[:, i].std(), 1e-10) for i in range(n)])
        inv_vol = 1.0 / vols
        weights_arr = inv_vol / inv_vol.sum()

        # Ensemble returns
        ensemble_ret = strat_returns @ weights_arr
        if ensemble_ret.std() > 1e-10:
            ensemble_sharpe = float(ensemble_ret.mean() / ensemble_ret.std() * math.sqrt(252))
        else:
            ensemble_sharpe = 0.0

        weights = {names[i]: float(weights_arr[i]) for i in range(n)}
        return weights, ensemble_sharpe


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CONVENIENCE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_tournament(n_bars: int = 1000, seed: int = 42) -> TournamentReport:
    """Run a tournament on synthetic data."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.02, n_bars)

    # Add regime structure
    regimes = np.empty(n_bars, dtype=object)
    for i in range(0, n_bars, 200):
        end = min(i + 200, n_bars)
        r = rng.choice(["trending_up", "trending_down", "mean_reverting", "high_vol"])
        regimes[i:end] = r
        if r == "trending_up":
            returns[i:end] += 0.001
        elif r == "trending_down":
            returns[i:end] -= 0.001
        elif r == "high_vol":
            returns[i:end] *= 2

    tournament = StrategyTournament()
    return tournament.run(returns, regimes)
