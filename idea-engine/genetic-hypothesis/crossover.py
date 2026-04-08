"""
Advanced crossover operators for genetic hypothesis evolution.

Implements:
  - Simulated binary crossover (SBX)
  - Order crossover (OX) for permutation problems
  - Semantic-preserving crossover (structure-aware)
  - Multi-point crossover
  - Adaptive crossover (self-adapting rates)
  - Semantic distance metric between chromosomes
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from idea_engine.genetic_hypothesis.evolver import Chromosome


# ── Simulated Binary Crossover (SBX) ─────────────────────────────────────────

def sbx_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    schema: dict,
    eta: float = 20.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Chromosome, Chromosome]:
    """
    Simulated Binary Crossover (Deb & Agrawal 1994).
    Mimics single-point binary crossover in continuous space.
    Higher eta = offspring closer to parents (less exploration).
    """
    rng = rng or np.random.default_rng()
    params_c, params_d = {}, {}

    for name, (lo, hi, dtype) in schema.items():
        x1 = float(parent_a.params[name])
        x2 = float(parent_b.params[name])

        if abs(x2 - x1) < 1e-10:
            params_c[name] = parent_a.params[name]
            params_d[name] = parent_b.params[name]
            continue

        if x1 > x2:
            x1, x2 = x2, x1

        u = rng.random()
        # Beta distribution
        beta = 1 + 2 * min(x1 - lo, hi - x2) / (x2 - x1)
        alpha = 2 - beta**(-(eta + 1))
        if u <= 1 / alpha:
            beta_q = (u * alpha)**(1 / (eta + 1))
        else:
            beta_q = (1 / (2 - u * alpha))**(1 / (eta + 1))

        c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
        c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
        params_c[name] = dtype(np.clip(c1, lo, hi))
        params_d[name] = dtype(np.clip(c2, lo, hi))

    return (
        Chromosome(params=params_c, parent_ids=[parent_a.id, parent_b.id]),
        Chromosome(params=params_d, parent_ids=[parent_a.id, parent_b.id]),
    )


# ── Blend Crossover (BLX-alpha) ───────────────────────────────────────────────

def blx_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    schema: dict,
    alpha: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Chromosome, Chromosome]:
    """
    BLX-alpha crossover: offspring in [x_min - alpha*d, x_max + alpha*d].
    alpha=0: convex combination only; alpha=0.5: most common.
    """
    rng = rng or np.random.default_rng()
    params_c, params_d = {}, {}

    for name, (lo, hi, dtype) in schema.items():
        x1 = float(parent_a.params[name])
        x2 = float(parent_b.params[name])
        x_min, x_max = min(x1, x2), max(x1, x2)
        d = x_max - x_min

        lb = max(x_min - alpha * d, lo)
        ub = min(x_max + alpha * d, hi)

        c1 = rng.uniform(lb, ub)
        c2 = rng.uniform(lb, ub)
        params_c[name] = dtype(c1)
        params_d[name] = dtype(c2)

    return (
        Chromosome(params=params_c, parent_ids=[parent_a.id, parent_b.id]),
        Chromosome(params=params_d, parent_ids=[parent_a.id, parent_b.id]),
    )


# ── Multi-Point Crossover ──────────────────────────────────────────────────────

def multipoint_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    schema: dict,
    n_points: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Chromosome, Chromosome]:
    """
    Multi-point crossover: split parameter list at n_points and alternate segments.
    """
    rng = rng or np.random.default_rng()
    keys = list(schema.keys())
    n = len(keys)

    # Choose crossover points
    points = sorted(rng.choice(n - 1, min(n_points, n - 1), replace=False) + 1)
    points = [0] + list(points) + [n]

    params_c, params_d = {}, {}
    use_a = True

    for i in range(len(points) - 1):
        segment = keys[points[i]: points[i + 1]]
        for k in segment:
            if use_a:
                params_c[k] = parent_a.params[k]
                params_d[k] = parent_b.params[k]
            else:
                params_c[k] = parent_b.params[k]
                params_d[k] = parent_a.params[k]
        use_a = not use_a

    return (
        Chromosome(params=params_c, parent_ids=[parent_a.id, parent_b.id]),
        Chromosome(params=params_d, parent_ids=[parent_a.id, parent_b.id]),
    )


# ── Semantic-Preserving Crossover ─────────────────────────────────────────────

def semantic_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    schema: dict,
    semantic_groups: dict[str, list[str]],
    rng: Optional[np.random.Generator] = None,
) -> tuple[Chromosome, Chromosome]:
    """
    Semantic-preserving crossover: exchange semantically coherent groups of parameters.
    E.g., {"entry": ["entry_z", "entry_vol_threshold"], "exit": ["exit_z", "stop_loss"]}
    Keeps entry/exit logic consistent within each offspring.
    """
    rng = rng or np.random.default_rng()
    params_c = parent_a.params.copy()
    params_d = parent_b.params.copy()

    for group_name, group_keys in semantic_groups.items():
        if rng.random() < 0.5:
            # Swap entire group
            for k in group_keys:
                if k in params_c and k in params_d:
                    params_c[k], params_d[k] = params_d[k], params_c[k]

    return (
        Chromosome(params=params_c, parent_ids=[parent_a.id, parent_b.id]),
        Chromosome(params=params_d, parent_ids=[parent_a.id, parent_b.id]),
    )


# ── Chromosome Distance ────────────────────────────────────────────────────────

def chromosome_distance(
    a: Chromosome,
    b: Chromosome,
    schema: dict,
) -> float:
    """
    Normalized Euclidean distance between two chromosomes in parameter space.
    Used for diversity maintenance and crowding.
    """
    total = 0.0
    for name, (lo, hi, _) in schema.items():
        if name in a.params and name in b.params:
            norm_a = (float(a.params[name]) - lo) / max(hi - lo, 1e-10)
            norm_b = (float(b.params[name]) - lo) / max(hi - lo, 1e-10)
            total += (norm_a - norm_b)**2
    return float(math.sqrt(total / max(len(schema), 1)))


# ── Adaptive Crossover ────────────────────────────────────────────────────────

class AdaptiveCrossover:
    """
    Self-adaptive crossover: adjusts operator probabilities based on offspring quality.
    Operators: SBX, BLX, multipoint, semantic.
    """

    def __init__(
        self,
        schema: dict,
        semantic_groups: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.schema = schema
        self.semantic_groups = semantic_groups or {}
        self.rng = rng or np.random.default_rng()
        self.operator_weights = {"sbx": 0.4, "blx": 0.3, "multipoint": 0.2, "semantic": 0.1}
        self.operator_successes = {k: 0 for k in self.operator_weights}
        self.operator_uses = {k: 1 for k in self.operator_weights}

    def crossover(
        self,
        parent_a: Chromosome,
        parent_b: Chromosome,
    ) -> tuple[Chromosome, Chromosome, str]:
        """
        Apply adaptively selected crossover operator.
        Returns (child1, child2, operator_name).
        """
        ops = list(self.operator_weights.keys())
        weights = np.array([self.operator_weights[op] for op in ops])
        weights /= weights.sum()
        op = ops[self.rng.choice(len(ops), p=weights)]

        self.operator_uses[op] += 1

        if op == "sbx":
            c1, c2 = sbx_crossover(parent_a, parent_b, self.schema, rng=self.rng)
        elif op == "blx":
            c1, c2 = blx_crossover(parent_a, parent_b, self.schema, rng=self.rng)
        elif op == "multipoint":
            c1, c2 = multipoint_crossover(parent_a, parent_b, self.schema, rng=self.rng)
        else:
            if self.semantic_groups:
                c1, c2 = semantic_crossover(parent_a, parent_b, self.schema, self.semantic_groups, rng=self.rng)
            else:
                c1, c2 = sbx_crossover(parent_a, parent_b, self.schema, rng=self.rng)
            op = "semantic"

        return c1, c2, op

    def update_reward(self, operator: str, success: bool) -> None:
        """Update operator weight based on offspring quality."""
        if success:
            self.operator_successes[operator] += 1
        # UCB1-style weight update
        for op in self.operator_weights:
            success_rate = self.operator_successes[op] / self.operator_uses[op]
            exploration = math.sqrt(2 * math.log(sum(self.operator_uses.values())) / self.operator_uses[op])
            self.operator_weights[op] = success_rate + 0.3 * exploration

    def diversity_stats(self, population: list[Chromosome]) -> dict:
        """Compute population diversity metrics."""
        n = len(population)
        if n < 2:
            return {"mean_distance": 0.0, "min_distance": 0.0}

        distances = []
        sample_size = min(n, 20)
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                d = chromosome_distance(population[i], population[j], self.schema)
                distances.append(d)

        return {
            "mean_distance": float(np.mean(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "diversity_index": float(np.mean(distances)),
        }
