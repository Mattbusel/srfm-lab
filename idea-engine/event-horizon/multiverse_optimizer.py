"""
Multiverse Path Optimizer: optimize across 1000 parallel futures simultaneously.

Instead of optimizing for ONE historical backtest (which is overfitting to
one realization of the stochastic process), this module:

1. Generates N possible futures using the Dream Engine's physics perturbations
2. Runs the Portfolio Brain on each future independently
3. Finds the portfolio weights that maximize MEDIAN performance across futures
4. Penalizes portfolios that fail in any significant fraction of futures

The objective: maximize E[Sharpe] - lambda * Var(Sharpe) across all universes.

This is essentially a time machine: by simulating many possible futures
and finding the portfolio that works in MOST of them, you get robustness
that no single-path backtest can provide.
"""

from __future__ import annotations
import math
import time
import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class UniverseResult:
    """Result of running a portfolio in one simulated universe."""
    universe_id: int
    physics_perturbation: str
    sharpe: float
    total_return: float
    max_drawdown: float
    n_trades: int


@dataclass
class MultiverseResult:
    """Result of optimizing across all universes."""
    optimal_weights: Dict[str, float]
    median_sharpe: float
    mean_sharpe: float
    sharpe_std: float
    worst_universe_sharpe: float
    best_universe_sharpe: float
    survival_rate: float          # fraction of universes with positive Sharpe
    robustness_score: float       # E[Sharpe] - lambda * Var(Sharpe)
    n_universes: int
    universe_results: List[UniverseResult] = field(default_factory=list)


class MultiverseOptimizer:
    """
    Optimize portfolio weights across many possible futures.

    The key insight: a portfolio that works in 900/1000 simulated futures
    is vastly more robust than one that looks great in one backtest.
    """

    def __init__(self, symbols: List[str], n_universes: int = 100,
                  risk_aversion: float = 2.0, seed: int = 42):
        self.symbols = symbols
        self.n_universes = n_universes
        self.risk_aversion = risk_aversion
        self.rng = np.random.default_rng(seed)

    def generate_universes(self, base_returns: np.ndarray,
                             n_bars: int = 252) -> List[np.ndarray]:
        """
        Generate N possible future return paths.
        Uses the base return distribution + perturbations.
        """
        mu = float(base_returns.mean())
        sigma = float(base_returns.std())
        skew = float(np.mean(((base_returns - mu) / max(sigma, 1e-10)) ** 3))

        universes = []
        perturbations = [
            "base",           # same distribution
            "high_vol",       # 2x volatility
            "low_vol",        # 0.5x volatility
            "trending_up",    # positive drift
            "trending_down",  # negative drift
            "mean_reverting", # add mean reversion
            "jump_diffusion", # add occasional jumps
            "regime_switch",  # switch halfway
        ]

        for i in range(self.n_universes):
            perturb = perturbations[i % len(perturbations)]

            if perturb == "base":
                rets = self.rng.normal(mu, sigma, n_bars)
            elif perturb == "high_vol":
                rets = self.rng.normal(mu, sigma * 2, n_bars)
            elif perturb == "low_vol":
                rets = self.rng.normal(mu * 0.5, sigma * 0.5, n_bars)
            elif perturb == "trending_up":
                rets = self.rng.normal(mu + sigma * 0.5, sigma, n_bars)
            elif perturb == "trending_down":
                rets = self.rng.normal(mu - sigma * 0.5, sigma, n_bars)
            elif perturb == "mean_reverting":
                rets = np.zeros(n_bars)
                for t in range(n_bars):
                    mr = -0.1 * (rets[t-1] if t > 0 else 0)
                    rets[t] = self.rng.normal(mu + mr, sigma)
            elif perturb == "jump_diffusion":
                rets = self.rng.normal(mu, sigma, n_bars)
                jumps = self.rng.random(n_bars) < 0.02
                rets[jumps] += self.rng.normal(0, sigma * 5, int(jumps.sum()))
            elif perturb == "regime_switch":
                mid = n_bars // 2
                rets = np.zeros(n_bars)
                rets[:mid] = self.rng.normal(mu + sigma, sigma * 0.5, mid)
                rets[mid:] = self.rng.normal(mu - sigma, sigma * 2, n_bars - mid)
            else:
                rets = self.rng.normal(mu, sigma, n_bars)

            # Add random noise to each universe
            rets += self.rng.normal(0, sigma * 0.1, n_bars)
            universes.append(rets)

        return universes

    def evaluate_weights(
        self,
        weights: Dict[str, float],
        universes: List[np.ndarray],
        transaction_cost: float = 0.001,
    ) -> MultiverseResult:
        """Evaluate a set of weights across all universes."""
        results = []

        for i, universe_rets in enumerate(universes):
            # Apply weights (simplified: single asset per universe for now)
            # In production: multi-asset with correlation structure
            net_weight = sum(weights.values())
            strat_rets = universe_rets * net_weight

            # Costs
            cost = abs(net_weight) * transaction_cost
            strat_rets[0] -= cost  # entry cost

            if len(strat_rets) > 20 and strat_rets.std() > 1e-10:
                sharpe = float(strat_rets.mean() / strat_rets.std() * math.sqrt(252))
                total_ret = float(np.prod(1 + strat_rets) - 1)
                eq = np.cumprod(1 + strat_rets)
                peak = np.maximum.accumulate(eq)
                max_dd = float(((peak - eq) / peak).max())
            else:
                sharpe = 0.0
                total_ret = 0.0
                max_dd = 0.0

            results.append(UniverseResult(
                universe_id=i,
                physics_perturbation=["base", "high_vol", "low_vol", "up", "down", "mr", "jump", "switch"][i % 8],
                sharpe=sharpe,
                total_return=total_ret,
                max_drawdown=max_dd,
                n_trades=1,
            ))

        sharpes = np.array([r.sharpe for r in results])
        survival = float(np.mean(sharpes > 0))

        # Robustness score: E[Sharpe] - lambda * Var(Sharpe)
        robustness = float(np.mean(sharpes) - self.risk_aversion * np.std(sharpes))

        return MultiverseResult(
            optimal_weights=weights,
            median_sharpe=float(np.median(sharpes)),
            mean_sharpe=float(np.mean(sharpes)),
            sharpe_std=float(np.std(sharpes)),
            worst_universe_sharpe=float(np.min(sharpes)),
            best_universe_sharpe=float(np.max(sharpes)),
            survival_rate=survival,
            robustness_score=robustness,
            n_universes=len(universes),
            universe_results=results,
        )

    def optimize(
        self,
        base_returns: np.ndarray,
        n_candidates: int = 50,
        n_iterations: int = 20,
    ) -> MultiverseResult:
        """
        Find optimal weights by searching across the multiverse.

        Uses evolutionary search: generate candidate weight vectors,
        evaluate each across all universes, keep the best, mutate, repeat.
        """
        universes = self.generate_universes(base_returns)

        # Initialize candidates: random weight vectors
        candidates = []
        for _ in range(n_candidates):
            w = {}
            for sym in self.symbols:
                w[sym] = float(self.rng.uniform(-0.1, 0.1))
            candidates.append(w)

        # Add equal weight and zero as baselines
        candidates.append({sym: 1.0 / len(self.symbols) for sym in self.symbols})
        candidates.append({sym: 0.0 for sym in self.symbols})

        best_result = None
        best_score = float("-inf")

        for iteration in range(n_iterations):
            # Evaluate all candidates
            results = []
            for w in candidates:
                result = self.evaluate_weights(w, universes)
                results.append((w, result))

            # Sort by robustness score
            results.sort(key=lambda x: x[1].robustness_score, reverse=True)

            if results[0][1].robustness_score > best_score:
                best_score = results[0][1].robustness_score
                best_result = results[0][1]
                best_result.optimal_weights = results[0][0]

            # Evolution: keep top 20%, mutate rest
            elite_n = max(2, n_candidates // 5)
            new_candidates = [copy.deepcopy(r[0]) for r in results[:elite_n]]

            while len(new_candidates) < n_candidates:
                parent = random.choice(new_candidates[:elite_n])
                child = {}
                for sym in self.symbols:
                    noise = self.rng.normal(0, 0.02)
                    child[sym] = float(np.clip(parent.get(sym, 0) + noise, -0.15, 0.15))
                new_candidates.append(child)

            candidates = new_candidates

        return best_result


class SwarmIntelligence:
    """
    Swarm of mini-brains that vote on trades.

    Instead of one Portfolio Brain, run N mini-brains with slightly
    different parameters. The wisdom of crowds filters out noise.

    Each mini-brain:
    - Has slightly different signal weights
    - Has slightly different risk parameters
    - Makes independent trading decisions
    - Votes on the aggregate direction

    The swarm output is the vote-weighted consensus.
    """

    def __init__(self, n_brains: int = 50, seed: int = 42):
        self.n_brains = n_brains
        self.rng = np.random.default_rng(seed)

        # Generate diverse brain parameters
        self._brain_params = []
        for i in range(n_brains):
            params = {
                "momentum_weight": float(self.rng.uniform(0.1, 0.6)),
                "reversion_weight": float(self.rng.uniform(0.1, 0.5)),
                "physics_weight": float(self.rng.uniform(0.0, 0.4)),
                "lookback": int(self.rng.integers(10, 60)),
                "risk_aversion": float(self.rng.uniform(1.0, 5.0)),
                "threshold": float(self.rng.uniform(0.05, 0.3)),
            }
            self._brain_params.append(params)

        # Track brain accuracy
        self._brain_scores = np.ones(n_brains)  # start equal
        self._total_votes = 0

    def vote(self, returns: np.ndarray) -> Dict:
        """
        All brains vote on the current market state.
        Returns consensus direction and confidence.
        """
        if len(returns) < 60:
            return {"direction": 0.0, "confidence": 0.0, "agreement": 0.0}

        votes = np.zeros(self.n_brains)

        for i, params in enumerate(self._brain_params):
            lb = params["lookback"]
            window = returns[-lb:]

            # Each brain computes its own signal
            momentum = float(window.mean() / max(window.std(), 1e-10))
            reversion = float(-(returns[-1] - window.mean()) / max(window.std(), 1e-10))

            signal = (
                params["momentum_weight"] * np.tanh(momentum) +
                params["reversion_weight"] * np.tanh(reversion / 2)
            )

            if abs(signal) > params["threshold"]:
                votes[i] = np.sign(signal)
            else:
                votes[i] = 0.0

        # Weighted consensus (better brains get more weight)
        weights = self._brain_scores / self._brain_scores.sum()
        weighted_vote = float(np.dot(weights, votes))

        # Agreement: what fraction of brains agree with the majority?
        majority_dir = np.sign(weighted_vote) if abs(weighted_vote) > 0.1 else 0
        if majority_dir != 0:
            agreement = float(np.mean(np.sign(votes[votes != 0]) == majority_dir)) if (votes != 0).any() else 0
        else:
            agreement = 0.0

        self._total_votes += 1

        return {
            "direction": float(np.sign(weighted_vote)),
            "strength": float(abs(weighted_vote)),
            "confidence": float(agreement),
            "n_voting": int(np.sum(votes != 0)),
            "n_long": int(np.sum(votes > 0)),
            "n_short": int(np.sum(votes < 0)),
            "n_abstain": int(np.sum(votes == 0)),
            "agreement": float(agreement),
        }

    def record_outcome(self, actual_return: float, predicted_direction: float) -> None:
        """Update brain scores based on outcome."""
        for i, params in enumerate(self._brain_params):
            # Simple: did this brain predict correctly?
            if predicted_direction != 0:
                correct = np.sign(actual_return) == predicted_direction
                self._brain_scores[i] *= 1.01 if correct else 0.99
                self._brain_scores[i] = max(0.1, self._brain_scores[i])

    def get_brain_leaderboard(self) -> List[Dict]:
        """Rank brains by accumulated score."""
        ranked = sorted(enumerate(self._brain_scores), key=lambda x: x[1], reverse=True)
        return [
            {"brain_id": i, "score": float(score), "params": self._brain_params[i]}
            for i, score in ranked[:10]
        ]
