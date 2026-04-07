"""
gp_operators.py -- Genetic operators for symbolic expression tree manipulation.

Provides crossover, mutation, and selection operators designed for GP trees.
All operators work on ExpressionTree objects and return new trees without
mutating the originals (unless otherwise noted for efficiency).

Selection operators:
  - fitness_proportional_select  (roulette wheel)
  - tournament_select
  - lexicase_select               (for multi-objective / regime-conditioned IC)

Crossover:
  - subtree_crossover             (standard GP crossover with depth guard)

Mutation:
  - point_mutation                (node-level replacement, arity-preserving)
  - subtree_mutation              (replace subtree with new random tree)
  - hoist_mutation                (promotes subtree -- reduces bloat)
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .expression_tree import (
    ExpressionNode,
    ExpressionTree,
    NodeType,
    TreeGenerator,
    ALL_FUNCTION_NAMES,
    ALL_TERMINALS,
    FUNCTION_ARITY,
    BINARY_FUNCTION_NAMES,
    UNARY_FUNCTION_NAMES,
    _EPS,
)


# ---------------------------------------------------------------------------
# Selection operators
# ---------------------------------------------------------------------------

def fitness_proportional_select(
    population: List[ExpressionTree],
    fitnesses: List[float],
) -> ExpressionTree:
    """
    Fitness-proportional (roulette wheel) selection.

    Fitnesses are shifted so the minimum is 0 before computing probabilities.
    If all fitnesses are identical, falls back to uniform random selection.
    """
    if len(population) == 1:
        return population[0]
    arr = np.array(fitnesses, dtype=np.float64)
    arr = arr - arr.min()
    total = arr.sum()
    if total < _EPS:
        return random.choice(population)
    probs = arr / total
    idx = int(np.random.choice(len(population), p=probs))
    return population[idx]


def tournament_select(
    population: List[ExpressionTree],
    fitnesses: List[float],
    k: int = 7,
) -> ExpressionTree:
    """
    Tournament selection.

    Randomly draws k individuals and returns the one with highest fitness.
    k is clamped to len(population).
    """
    n = len(population)
    k = min(k, n)
    indices = random.sample(range(n), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def lexicase_select(
    population: List[ExpressionTree],
    fitnesses_matrix: np.ndarray,
) -> ExpressionTree:
    """
    Lexicase selection for multi-objective / case-based fitness.

    fitnesses_matrix: shape (population_size, n_cases)
      Each column is one fitness case (e.g., IC on a particular regime slice).

    Algorithm:
      1. Start with entire population as candidates.
      2. Shuffle fitness cases.
      3. For each case in shuffled order:
         a. Find the best performance on that case among current candidates.
         b. Keep only candidates within epsilon of best (median absolute deviation
            of the case distribution is used as epsilon).
      4. Return a random survivor from the final candidate set.
    """
    n_pop, n_cases = fitnesses_matrix.shape
    if n_pop == 0:
        raise ValueError("Empty population")
    if n_pop == 1:
        return population[0]

    candidates = list(range(n_pop))
    case_order = list(range(n_cases))
    random.shuffle(case_order)

    for case_idx in case_order:
        if len(candidates) == 1:
            break
        case_fitnesses = fitnesses_matrix[candidates, case_idx]
        best_val = float(np.nanmax(case_fitnesses))
        # adaptive epsilon: median absolute deviation of the full column
        col_all = fitnesses_matrix[:, case_idx]
        col_valid = col_all[~np.isnan(col_all)]
        if len(col_valid) > 0:
            mad = float(np.median(np.abs(col_valid - np.median(col_valid))))
            epsilon = max(mad * 0.1, _EPS)
        else:
            epsilon = _EPS
        candidates = [
            c for c, fv in zip(candidates, case_fitnesses)
            if not np.isnan(fv) and fv >= best_val - epsilon
        ]
        if len(candidates) == 0:
            # fallback: restore previous best
            best_c_idx = int(np.nanargmax(fitnesses_matrix[:, case_idx]))
            candidates = [best_c_idx]
            break

    chosen = random.choice(candidates)
    return population[chosen]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def subtree_crossover(
    tree1: ExpressionTree,
    tree2: ExpressionTree,
    max_depth: int = 8,
) -> Tuple[ExpressionTree, ExpressionTree]:
    """
    Standard GP subtree crossover.

    Selects a random node in each parent, swaps the subtrees rooted there.
    If either offspring exceeds max_depth, the original parents are returned
    unchanged (rejecting the crossover).

    Returns a tuple (offspring1, offspring2).
    """
    child1 = tree1.copy()
    child2 = tree2.copy()

    # pick random crossover points
    node1, parent1, side1 = child1.get_subtree_root_at_random()
    node2, parent2, side2 = child2.get_subtree_root_at_random()

    # swap: put node2's subtree into child1, node1's subtree into child2
    node1_copy = copy.deepcopy(node1)
    node2_copy = copy.deepcopy(node2)

    child1.set_subtree(parent1, side1, node2_copy)
    child2.set_subtree(parent2, side2, node1_copy)

    # depth guard -- reject if bloated
    if child1.depth() > max_depth or child2.depth() > max_depth:
        return tree1.copy(), tree2.copy()

    return child1, child2


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def point_mutation(
    tree: ExpressionTree,
    rate: float = 0.05,
    generator: Optional[TreeGenerator] = None,
) -> ExpressionTree:
    """
    Point mutation: each node is independently mutated with probability `rate`.

    Mutation rules:
    -- TERMINAL  -> different terminal chosen uniformly (same arity: 0)
    -- CONSTANT  -> Gaussian perturbation N(0, 0.5) added; clamped to [-10, 10]
    -- FUNCTION  -> replaced by different function of SAME arity
                    (to preserve tree structure)
    """
    if generator is None:
        generator = TreeGenerator()
    mutated = tree.copy()
    nodes = mutated.collect_nodes()
    for node, parent, side in nodes:
        if random.random() > rate:
            continue
        if node.node_type == NodeType.TERMINAL:
            new_name = random.choice(generator.terminal_names)
            node.value = new_name
        elif node.node_type == NodeType.CONSTANT:
            perturbation = random.gauss(0.0, 0.5)
            node.value = float(np.clip(node.value + perturbation, -10.0, 10.0))
        elif node.node_type == NodeType.FUNCTION:
            current_arity = node.arity
            if current_arity == 1:
                candidates = [n for n in UNARY_FUNCTION_NAMES if n != node.value]
            else:
                candidates = [n for n in BINARY_FUNCTION_NAMES if n != node.value]
            if candidates:
                new_fn = random.choice(candidates)
                node.value = new_fn
                node.arity = FUNCTION_ARITY[new_fn]
    return mutated


def subtree_mutation(
    tree: ExpressionTree,
    max_depth: int = 3,
    generator: Optional[TreeGenerator] = None,
) -> ExpressionTree:
    """
    Subtree mutation: select a random node and replace its subtree with a
    newly generated random tree of depth <= max_depth.

    Uses the grow method for the replacement subtree.
    """
    if generator is None:
        generator = TreeGenerator()
    mutated = tree.copy()
    node, parent, side = mutated.get_subtree_root_at_random()
    new_subtree_tree = generator.random_tree(max_depth=max_depth, method="grow")
    mutated.set_subtree(parent, side, new_subtree_tree.root)
    return mutated


def hoist_mutation(tree: ExpressionTree) -> ExpressionTree:
    """
    Hoist mutation: select a random subtree, then select a random subtree
    within THAT subtree, and replace the original subtree with the inner one.

    This reduces tree depth (anti-bloat pressure).
    If the tree is just a leaf, returns an unchanged copy.
    """
    mutated = tree.copy()
    nodes = mutated.collect_nodes()
    # prefer internal (function) nodes to hoist from
    internal = [(n, p, s) for n, p, s in nodes
                if n.node_type == NodeType.FUNCTION]
    if not internal:
        return mutated

    # pick an internal node as the outer subtree
    outer_node, outer_parent, outer_side = random.choice(internal)

    # collect all nodes within outer subtree (including outer_node itself)
    temp_tree = ExpressionTree(outer_node)
    inner_nodes = temp_tree.collect_nodes()
    if len(inner_nodes) <= 1:
        return mutated

    # pick a different node inside to hoist up
    # exclude outer_node itself (side == 'root') to avoid no-op
    hoistable = [(n, p, s) for n, p, s in inner_nodes if s != "root"]
    if not hoistable:
        return mutated

    inner_node, _, _ = random.choice(hoistable)
    inner_copy = copy.deepcopy(inner_node)

    # replace outer subtree with the inner node
    mutated.set_subtree(outer_parent, outer_side, inner_copy)
    return mutated


# ---------------------------------------------------------------------------
# Operator selection utility
# ---------------------------------------------------------------------------

def apply_random_mutation(
    tree: ExpressionTree,
    point_rate: float = 0.05,
    max_subtree_depth: int = 3,
    max_tree_depth: int = 8,
    generator: Optional[TreeGenerator] = None,
) -> ExpressionTree:
    """
    Apply one of the three mutation operators at random:
      - 40% subtree mutation
      - 40% point mutation
      - 20% hoist mutation

    After mutation, if depth exceeds max_tree_depth, apply hoist until
    depth is acceptable (up to 5 attempts).
    """
    if generator is None:
        generator = TreeGenerator()

    r = random.random()
    if r < 0.40:
        result = subtree_mutation(tree, max_depth=max_subtree_depth, generator=generator)
    elif r < 0.80:
        result = point_mutation(tree, rate=point_rate, generator=generator)
    else:
        result = hoist_mutation(tree)

    # bloat control
    attempts = 0
    while result.depth() > max_tree_depth and attempts < 5:
        result = hoist_mutation(result)
        attempts += 1

    return result
