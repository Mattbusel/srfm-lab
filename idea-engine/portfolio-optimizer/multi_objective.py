"""
portfolio-optimizer/multi_objective.py

Multi-objective portfolio optimisation using NSGA-II style evolutionary
search and Pareto-front construction.

Objectives (all evaluated in maximisation form internally):
  1. Maximise Sharpe ratio
  2. Maximise Calmar ratio
  3. Minimise maximum drawdown (maximise negative MaxDD)
  4. Maximise Sortino ratio

The Pareto front contains the set of non-dominated portfolios across all
four objectives.  A preferred portfolio is selected from the front using
a risk-tolerance-weighted scalarisation.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParetoPortfolio:
    """
    A single Pareto-optimal portfolio solution.

    Attributes
    ----------
    portfolio_id : str
        Unique identifier.
    weights : np.ndarray
        Asset weights summing to 1.0.
    asset_names : list[str]
        Asset names for each weight element.
    sharpe : float
        Estimated Sharpe ratio.
    calmar : float
        Estimated Calmar ratio.
    max_drawdown : float
        Estimated maximum drawdown (positive fraction).
    sortino : float
        Estimated Sortino ratio.
    scalarised_score : float
        Weighted combination of objectives after scalarisation.
    rank : int
        Pareto front rank (0 = truly Pareto-optimal).
    crowding_distance : float
        NSGA-II crowding distance for diversity preservation.
    """

    portfolio_id: str
    weights: np.ndarray
    asset_names: list[str]
    sharpe: float
    calmar: float
    max_drawdown: float
    sortino: float
    scalarised_score: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.asset_names, self.weights.tolist()))

    def objectives_vector(self) -> np.ndarray:
        """Return objectives as a maximisation vector (MaxDD is negated)."""
        return np.array([self.sharpe, self.calmar, -self.max_drawdown, self.sortino])

    def summary(self) -> str:
        lines = [
            f"Portfolio {self.portfolio_id[:8]}",
            f"  Sharpe       : {self.sharpe:.4f}",
            f"  Calmar       : {self.calmar:.4f}",
            f"  MaxDD        : {self.max_drawdown:.4f}",
            f"  Sortino      : {self.sortino:.4f}",
            f"  Score        : {self.scalarised_score:.4f}",
            f"  Pareto rank  : {self.rank}",
            "  Weights:",
        ]
        for name, w in zip(self.asset_names, self.weights):
            lines.append(f"    {name:<20} {w:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core multi-objective optimiser
# ---------------------------------------------------------------------------


class MultiObjectivePortfolioOptimizer:
    """
    Multi-objective portfolio optimiser using NSGA-II style evolutionary search.

    Parameters
    ----------
    expected_returns : np.ndarray
        Per-bar expected returns for each asset.
    cov_matrix : np.ndarray
        Asset covariance matrix.
    asset_names : list[str], optional
        Asset names.
    annualisation : int
        Bars per year.
    rf : float
        Risk-free rate per bar.
    allow_short : bool
        Allow negative weights.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> mu = rng.normal(0.0005, 0.0002, 6)
    >>> sigma = rng.normal(0.012, 0.002, 6)
    >>> cov = np.diag(sigma**2)
    >>> opt = MultiObjectivePortfolioOptimizer(mu, cov)
    >>> front = opt.pareto_front(n_points=20)
    >>> best = opt.preferred_portfolio(front, risk_tolerance=0.5)
    >>> print(best.summary())
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[list[str]] = None,
        annualisation: int = 252,
        rf: float = 0.0,
        allow_short: bool = False,
    ) -> None:
        self.expected_returns = np.asarray(expected_returns, dtype=float)
        self.cov_matrix = np.asarray(cov_matrix, dtype=float)
        n = len(self.expected_returns)
        self.asset_names = asset_names or [f"asset_{i}" for i in range(n)]
        self.annualisation = annualisation
        self.rf = rf
        self.allow_short = allow_short
        self._rng = np.random.default_rng(seed=0)

    # ------------------------------------------------------------------
    # Pareto front construction
    # ------------------------------------------------------------------

    def pareto_front(
        self,
        n_points: int = 50,
    ) -> list[ParetoPortfolio]:
        """
        Build the Pareto front by evaluating ``n_points`` candidate portfolios
        drawn from grid sampling + local optimisation for each objective.

        Parameters
        ----------
        n_points : int
            Number of candidate portfolios to evaluate.

        Returns
        -------
        list[ParetoPortfolio]
            Non-dominated portfolios on the Pareto front (rank=0), sorted
            by decreasing scalarised score.
        """
        candidates: list[ParetoPortfolio] = []

        # 1. Random portfolios
        n_random = max(n_points * 3, 100)
        for _ in range(n_random):
            w = self._random_weight()
            p = self._evaluate(w)
            candidates.append(p)

        # 2. Objective-specific local optima
        for objective_fn in [
            self._neg_sharpe,
            self._neg_calmar,
            self._max_dd_objective,
            self._neg_sortino,
        ]:
            for _ in range(5):
                w0 = self._random_weight()
                w_opt = self._local_optimize(objective_fn, w0)
                candidates.append(self._evaluate(w_opt))

        # 3. NSGA-II style evolutionary search
        evo_candidates = self.evolutionary_optimize(
            n_pop=max(n_points, 30), n_gen=20
        )
        candidates.extend(evo_candidates)

        # 4. Non-dominated sorting
        candidates = self._fast_non_dominated_sort(candidates)

        # Keep only Pareto front (rank=0)
        front = [p for p in candidates if p.rank == 0]

        # 5. Crowding distance
        self._compute_crowding_distance(front)

        # 6. Scalarise for default ordering
        front_scored = [
            self._apply_scalarisation(p, weights=(0.4, 0.3, 0.3))
            for p in front
        ]
        front_scored.sort(key=lambda p: p.scalarised_score, reverse=True)

        return front_scored

    # ------------------------------------------------------------------
    # Scalarisation
    # ------------------------------------------------------------------

    def scalarize(
        self,
        objectives: np.ndarray,
        weights: tuple[float, ...] = (0.4, 0.3, 0.3),
    ) -> float:
        """
        Scalarise the objective vector to a single score.

        The four objectives (Sharpe, Calmar, -MaxDD, Sortino) are normalised
        by their approximate ranges then combined with the provided weights.
        The weight vector is broadcast as:
          w[0] → Sharpe
          w[1] → Calmar  (or -MaxDD if 3 weights)
          w[2] → Sortino (or Calmar if 4 weights)
          w[3] → -MaxDD  (if 4 weights provided)

        Parameters
        ----------
        objectives : np.ndarray
            [sharpe, calmar, -max_dd, sortino] vector.
        weights : tuple[float, ...]
            Objective weights.  If 3 provided, Sortino is dropped.

        Returns
        -------
        float
            Weighted scalarised score.
        """
        obj = np.asarray(objectives, dtype=float)
        # Clip to avoid extreme values from small samples
        obj = np.clip(obj, -10.0, 10.0)

        if len(weights) == 3:
            # Use sharpe, -max_dd, sortino
            selected = np.array([obj[0], obj[2], obj[3]])
            w = np.array(weights) / sum(weights)
        elif len(weights) == 4:
            selected = obj
            w = np.array(weights) / sum(weights)
        else:
            raise ValueError(f"Expected 3 or 4 weights, got {len(weights)}")

        return float(w @ selected[: len(w)])

    # ------------------------------------------------------------------
    # Evolutionary optimisation (NSGA-II inspired)
    # ------------------------------------------------------------------

    def evolutionary_optimize(
        self,
        n_pop: int = 100,
        n_gen: int = 50,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        mutation_scale: float = 0.05,
    ) -> list[ParetoPortfolio]:
        """
        NSGA-II inspired evolutionary optimisation of portfolio weights.

        Uses:
          - Tournament selection (binary tournament on Pareto rank + crowding)
          - Simulated binary crossover (SBX) on weight vectors
          - Gaussian mutation with simplex projection

        Parameters
        ----------
        n_pop : int
            Population size.
        n_gen : int
            Number of generations.
        crossover_prob : float
            Probability of applying crossover.
        mutation_prob : float
            Probability of mutating each gene.
        mutation_scale : float
            Standard deviation of Gaussian mutation noise.

        Returns
        -------
        list[ParetoPortfolio]
            Final population of evaluated portfolio solutions.
        """
        n = len(self.expected_returns)

        # Initialise population
        population: list[np.ndarray] = [self._random_weight() for _ in range(n_pop)]
        evaluated: list[ParetoPortfolio] = [self._evaluate(w) for w in population]

        for gen in range(n_gen):
            # Non-dominated sort + crowding for selection
            evaluated = self._fast_non_dominated_sort(evaluated)
            self._compute_crowding_distance(evaluated)

            offspring_weights: list[np.ndarray] = []

            while len(offspring_weights) < n_pop:
                # Tournament selection
                p1 = self._tournament_select(evaluated)
                p2 = self._tournament_select(evaluated)

                # Crossover
                if self._rng.random() < crossover_prob:
                    c1, c2 = self._sbx_crossover(p1.weights, p2.weights)
                else:
                    c1, c2 = p1.weights.copy(), p2.weights.copy()

                # Mutation
                c1 = self._mutate(c1, mutation_prob, mutation_scale)
                c2 = self._mutate(c2, mutation_prob, mutation_scale)

                offspring_weights.append(c1)
                if len(offspring_weights) < n_pop:
                    offspring_weights.append(c2)

            offspring = [self._evaluate(w) for w in offspring_weights]

            # Combine parent + offspring, keep top n_pop
            combined = evaluated + offspring
            combined = self._fast_non_dominated_sort(combined)
            self._compute_crowding_distance(combined)

            # Survival: prefer lower rank, then higher crowding distance
            combined.sort(key=lambda p: (p.rank, -p.crowding_distance))
            evaluated = combined[:n_pop]

        return evaluated

    # ------------------------------------------------------------------
    # Preferred portfolio selection
    # ------------------------------------------------------------------

    def preferred_portfolio(
        self,
        pareto_front: list[ParetoPortfolio],
        risk_tolerance: float = 0.5,
    ) -> ParetoPortfolio:
        """
        Select the preferred portfolio from the Pareto front based on
        a risk-tolerance parameter.

        risk_tolerance=0.0 → most conservative (lowest MaxDD, highest Calmar).
        risk_tolerance=1.0 → most aggressive (highest Sharpe, highest Sortino).
        risk_tolerance=0.5 → balanced (default scalarisation).

        Parameters
        ----------
        pareto_front : list[ParetoPortfolio]
            Output of :meth:`pareto_front`.
        risk_tolerance : float
            Value in [0, 1].

        Returns
        -------
        ParetoPortfolio
            The preferred portfolio.
        """
        if not pareto_front:
            raise ValueError("Pareto front is empty")

        rt = float(np.clip(risk_tolerance, 0.0, 1.0))

        # Blend objective weights based on risk tolerance
        # High tolerance: care more about Sharpe and Sortino
        # Low tolerance:  care more about MaxDD and Calmar
        sharpe_w = 0.20 + 0.40 * rt
        sortino_w = 0.10 + 0.30 * rt
        calmar_w = 0.30 - 0.20 * rt
        maxdd_w = 0.40 - 0.10 * rt  # -MaxDD objective

        obj_weights = (sharpe_w, calmar_w, maxdd_w, sortino_w)

        scored = [
            self._apply_scalarisation(p, weights=obj_weights)
            for p in pareto_front
        ]
        return max(scored, key=lambda p: p.scalarised_score)

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------

    def _evaluate(self, weights: np.ndarray) -> ParetoPortfolio:
        """Evaluate a weight vector and return a ParetoPortfolio."""
        w = np.asarray(weights, dtype=float)
        w = np.clip(w, 0.0 if not self.allow_short else -1.0, 1.0)
        total = w.sum()
        if total == 0:
            w = np.ones(len(w)) / len(w)
        else:
            w = w / total

        sharpe = self._compute_sharpe(w)
        calmar, max_dd = self._compute_calmar(w)
        sortino = self._compute_sortino(w)

        return ParetoPortfolio(
            portfolio_id=str(uuid.uuid4()),
            weights=w,
            asset_names=self.asset_names,
            sharpe=sharpe,
            calmar=calmar,
            max_drawdown=max_dd,
            sortino=sortino,
        )

    def _compute_sharpe(self, w: np.ndarray) -> float:
        mu = float(self.expected_returns @ w) * self.annualisation
        vol = float(np.sqrt(w @ self.cov_matrix @ w)) * np.sqrt(self.annualisation)
        if vol == 0:
            return 0.0
        return float((mu - self.rf * self.annualisation) / vol)

    def _compute_calmar(self, w: np.ndarray) -> tuple[float, float]:
        """Approximate Calmar via analytical max-drawdown estimate."""
        mu = float(self.expected_returns @ w) * self.annualisation
        vol = float(np.sqrt(w @ self.cov_matrix @ w)) * np.sqrt(self.annualisation)
        # Analytical approximation: E[MaxDD] ≈ vol * sqrt(T) for GBM
        # Here T = 1 year
        max_dd_approx = float(np.clip(vol * 0.5, 0.01, 1.0))  # simplified
        calmar = mu / max_dd_approx if max_dd_approx > 0 else 0.0
        return float(calmar), float(max_dd_approx)

    def _compute_sortino(self, w: np.ndarray) -> float:
        mu = float(self.expected_returns @ w) * self.annualisation
        # Downside vol: approximate via semi-variance
        # For normal distribution: downside_std ≈ vol * sqrt(0.5/π)
        vol = float(np.sqrt(w @ self.cov_matrix @ w)) * np.sqrt(self.annualisation)
        downside_vol = vol * np.sqrt(0.5 / np.pi)
        if downside_vol == 0:
            return 0.0
        return float(mu / downside_vol)

    def _neg_sharpe(self, w: np.ndarray) -> float:
        return -self._compute_sharpe(self._normalise(w))

    def _neg_calmar(self, w: np.ndarray) -> float:
        calmar, _ = self._compute_calmar(self._normalise(w))
        return -calmar

    def _max_dd_objective(self, w: np.ndarray) -> float:
        _, max_dd = self._compute_calmar(self._normalise(w))
        return max_dd  # minimise max_dd

    def _neg_sortino(self, w: np.ndarray) -> float:
        return -self._compute_sortino(self._normalise(w))

    # ------------------------------------------------------------------
    # NSGA-II helpers
    # ------------------------------------------------------------------

    def _fast_non_dominated_sort(
        self, portfolios: list[ParetoPortfolio]
    ) -> list[ParetoPortfolio]:
        """
        Assign Pareto ranks using fast non-dominated sort (Deb et al. 2002).

        Portfolios on the true Pareto front receive rank=0; portfolios that
        are dominated only by rank-0 portfolios receive rank=1, and so on.
        """
        n = len(portfolios)
        if n == 0:
            return portfolios

        obj_matrix = np.array([p.objectives_vector() for p in portfolios])

        domination_count = np.zeros(n, dtype=int)  # how many dominate me
        dominated_by_me: list[list[int]] = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(obj_matrix[i], obj_matrix[j]):
                    dominated_by_me[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(obj_matrix[j], obj_matrix[i]):
                    dominated_by_me[j].append(i)
                    domination_count[i] += 1

        front_0 = [i for i in range(n) if domination_count[i] == 0]
        current_front = front_0
        rank = 0

        while current_front:
            for i in current_front:
                portfolios[i].rank = rank
            next_front = []
            for i in current_front:
                for j in dominated_by_me[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            rank += 1

        return portfolios

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """Return True if solution a Pareto-dominates solution b."""
        return bool(np.all(a >= b) and np.any(a > b))

    def _compute_crowding_distance(
        self, portfolios: list[ParetoPortfolio]
    ) -> None:
        """Compute and assign NSGA-II crowding distances in place."""
        n = len(portfolios)
        if n == 0:
            return

        for p in portfolios:
            p.crowding_distance = 0.0

        n_obj = 4
        obj_matrix = np.array([p.objectives_vector() for p in portfolios])

        for m in range(n_obj):
            sorted_idx = np.argsort(obj_matrix[:, m])
            obj_range = obj_matrix[sorted_idx[-1], m] - obj_matrix[sorted_idx[0], m]

            # Boundary portfolios get infinite distance
            portfolios[sorted_idx[0]].crowding_distance = np.inf
            portfolios[sorted_idx[-1]].crowding_distance = np.inf

            for k in range(1, n - 1):
                if obj_range > 0:
                    portfolios[sorted_idx[k]].crowding_distance += (
                        obj_matrix[sorted_idx[k + 1], m]
                        - obj_matrix[sorted_idx[k - 1], m]
                    ) / obj_range

    def _tournament_select(
        self, population: list[ParetoPortfolio]
    ) -> ParetoPortfolio:
        """Binary tournament selection on rank + crowding distance."""
        i, j = self._rng.integers(0, len(population), size=2)
        a, b = population[i], population[j]
        if a.rank < b.rank:
            return a
        if b.rank < a.rank:
            return b
        if a.crowding_distance > b.crowding_distance:
            return a
        return b

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        eta: float = 15.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX) for real-valued weights.

        Parameters
        ----------
        parent1, parent2 : np.ndarray
            Parent weight vectors.
        eta : float
            Distribution index controlling offspring spread.

        Returns
        -------
        (child1, child2) : normalised child weight vectors.
        """
        n = len(parent1)
        c1 = parent1.copy()
        c2 = parent2.copy()

        for i in range(n):
            if self._rng.random() > 0.5:
                continue
            u = self._rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))

            c1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            c2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

        c1 = np.clip(c1, 0.0, 1.0)
        c2 = np.clip(c2, 0.0, 1.0)
        return self._normalise(c1), self._normalise(c2)

    def _mutate(
        self,
        weights: np.ndarray,
        prob: float,
        scale: float,
    ) -> np.ndarray:
        """Gaussian mutation with simplex projection."""
        mask = self._rng.random(len(weights)) < prob
        noise = self._rng.normal(0, scale, len(weights))
        w = weights + mask * noise
        w = np.clip(w, 0.0, 1.0)
        return self._normalise(w)

    # ------------------------------------------------------------------
    # Local optimisation
    # ------------------------------------------------------------------

    def _local_optimize(
        self,
        objective: Callable[[np.ndarray], float],
        w0: np.ndarray,
    ) -> np.ndarray:
        """Run SLSQP to locally minimise an objective from a starting point."""
        n = len(w0)
        lb = -1.0 if self.allow_short else 0.0
        bounds = Bounds(lb=lb, ub=1.0)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 500, "disp": False},
        )
        w = np.clip(res.x, 0.0, 1.0)
        return self._normalise(w)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_weight(self) -> np.ndarray:
        """Generate a random Dirichlet-distributed weight vector."""
        n = len(self.expected_returns)
        w = self._rng.dirichlet(np.ones(n))
        return w.astype(float)

    @staticmethod
    def _normalise(w: np.ndarray) -> np.ndarray:
        total = w.sum()
        if total == 0:
            n = len(w)
            return np.ones(n) / n
        return w / total

    def _apply_scalarisation(
        self,
        portfolio: ParetoPortfolio,
        weights: tuple[float, ...],
    ) -> ParetoPortfolio:
        """Compute and assign the scalarised score to a portfolio."""
        score = self.scalarize(portfolio.objectives_vector(), weights)
        portfolio.scalarised_score = score
        return portfolio
