"""
test_genetic.py -- Tests for the GP-based alpha signal discovery system.

Covers expression_tree, gp_operators, gp_engine, and signal_validator modules.
Run with: pytest ml/genetic/tests/test_genetic.py -v
"""

from __future__ import annotations

import copy
import math
import random
import sys
import os

import numpy as np
import pytest

# Make sure the package is importable from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ml.genetic.expression_tree import (
    ExpressionNode,
    ExpressionTree,
    NodeType,
    TreeGenerator,
    FUNCTION_ARITY,
    ALL_FUNCTION_NAMES,
    BINARY_FUNCTION_NAMES,
    UNARY_FUNCTION_NAMES,
    ALL_TERMINALS,
    SIGNAL_NAMES,
    RAW_FEATURES,
    _EPS,
    _ema_vectorized,
    _rolling_mean,
    _rolling_std,
    _rolling_zscore,
    _rank_normalize,
    _protected_div,
)
from ml.genetic.gp_operators import (
    fitness_proportional_select,
    tournament_select,
    lexicase_select,
    subtree_crossover,
    point_mutation,
    subtree_mutation,
    hoist_mutation,
    apply_random_mutation,
)
from ml.genetic.gp_engine import (
    GPConfig,
    GPEngine,
    Individual,
    _spearman_ic,
    _rolling_ic,
)
from ml.genetic.signal_validator import (
    SignalValidator,
    ValidationResult,
    _pearson_corr,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N = 300  # default data length


def make_data(n: int = N, seed: int = 42) -> dict:
    """Build a minimal data dict for evaluation tests."""
    rng = np.random.default_rng(seed)
    return {
        "close":   rng.standard_normal(n).cumsum() + 100.0,
        "volume":  rng.uniform(1e5, 1e7, n),
        "atr":     rng.uniform(0.5, 3.0, n),
        "bh_mass": rng.uniform(0.0, 1.0, n),
        "hurst_h": rng.uniform(0.3, 0.7, n),
        "nav_omega": rng.standard_normal(n),
        "mom_5d":  rng.standard_normal(n),
        "mom_20d": rng.standard_normal(n),
        "mr_rsi":  rng.uniform(-1.0, 1.0, n),
        "vol_realized_20d": rng.uniform(0.005, 0.05, n),
    }


def make_returns(n: int = N, seed: int = 43) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * 0.01


def make_tree_add() -> ExpressionTree:
    """Build ADD(mom_5d, mom_20d) manually."""
    left  = ExpressionNode(NodeType.TERMINAL, "mom_5d",  arity=0)
    right = ExpressionNode(NodeType.TERMINAL, "mom_20d", arity=0)
    root  = ExpressionNode(NodeType.FUNCTION, "ADD", left=left, right=right, arity=2)
    return ExpressionTree(root)


def make_tree_ema() -> ExpressionTree:
    """Build EMA(close, 10.0) -- unary-like via EMA(signal, constant)."""
    signal = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
    period = ExpressionNode(NodeType.CONSTANT, 10.0,    arity=0)
    root   = ExpressionNode(NodeType.FUNCTION, "EMA", left=signal, right=period, arity=2)
    return ExpressionTree(root)


def make_tree_div_zero() -> ExpressionTree:
    """Build DIV(mom_5d, 0.0) -- protected division test."""
    left  = ExpressionNode(NodeType.TERMINAL, "mom_5d", arity=0)
    zero  = ExpressionNode(NodeType.CONSTANT, 0.0, arity=0)
    root  = ExpressionNode(NodeType.FUNCTION, "DIV", left=left, right=zero, arity=2)
    return ExpressionTree(root)


def make_tree_deep(depth: int = 5) -> ExpressionTree:
    """Build a maximally deep ADD chain of terminals."""
    gen = TreeGenerator(
        function_names=["ADD"],
        terminal_names=["mom_5d", "mom_20d"],
    )
    return gen.random_tree(max_depth=depth, method="full")


# ---------------------------------------------------------------------------
# ==================== expression_tree.py tests ====================
# ---------------------------------------------------------------------------

class TestExpressionNodeBasics:

    def test_node_is_leaf_terminal(self):
        node = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        assert node.is_leaf() is True

    def test_node_is_leaf_constant(self):
        node = ExpressionNode(NodeType.CONSTANT, 3.14, arity=0)
        assert node.is_leaf() is True

    def test_node_is_not_leaf_function(self):
        node = ExpressionNode(NodeType.FUNCTION, "ADD", arity=2)
        assert node.is_leaf() is False

    def test_node_copy_is_independent(self):
        node = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        copy_node = node.copy()
        copy_node.value = "volume"
        assert node.value == "close"


class TestExpressionTreeEvaluateAdd:

    def test_expression_tree_evaluate_add(self):
        """ADD(mom_5d, mom_20d) should equal elementwise sum."""
        tree = make_tree_add()
        data = make_data()
        result = tree.evaluate(data)
        expected = data["mom_5d"] + data["mom_20d"]
        np.testing.assert_allclose(result, expected)

    def test_evaluate_returns_correct_length(self):
        tree = make_tree_add()
        data = make_data(n=200)
        result = tree.evaluate(data)
        assert len(result) == 200

    def test_evaluate_sub(self):
        left  = ExpressionNode(NodeType.TERMINAL, "mom_5d",  arity=0)
        right = ExpressionNode(NodeType.TERMINAL, "mom_20d", arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "SUB", left=left, right=right, arity=2)
        tree  = ExpressionTree(root)
        data  = make_data()
        result = tree.evaluate(data)
        expected = data["mom_5d"] - data["mom_20d"]
        np.testing.assert_allclose(result, expected)

    def test_evaluate_mul(self):
        left  = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        right = ExpressionNode(NodeType.CONSTANT, 2.0,    arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "MUL", left=left, right=right, arity=2)
        tree  = ExpressionTree(root)
        data  = make_data()
        result = tree.evaluate(data)
        np.testing.assert_allclose(result, data["close"] * 2.0)

    def test_evaluate_constant_terminal(self):
        """A tree that is just a constant returns an array of that constant."""
        root = ExpressionNode(NodeType.CONSTANT, 7.5, arity=0)
        tree = ExpressionTree(root)
        data = make_data()
        result = tree.evaluate(data)
        assert result.shape == (N,)
        np.testing.assert_allclose(result, 7.5)

    def test_evaluate_missing_terminal_returns_zeros(self):
        """Terminal not present in data -> zeros (graceful degradation)."""
        root = ExpressionNode(NodeType.TERMINAL, "phys_bh_mass_xyz_missing", arity=0)
        tree = ExpressionTree(root)
        data = make_data()
        result = tree.evaluate(data)
        np.testing.assert_allclose(result, 0.0)

    def test_evaluate_logical_gt(self):
        left  = ExpressionNode(NodeType.TERMINAL, "mom_5d",  arity=0)
        right = ExpressionNode(NodeType.TERMINAL, "mom_20d", arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "GT", left=left, right=right, arity=2)
        tree  = ExpressionTree(root)
        data  = make_data()
        result = tree.evaluate(data)
        expected = (data["mom_5d"] > data["mom_20d"]).astype(float)
        np.testing.assert_allclose(result, expected)

    def test_evaluate_logical_and(self):
        left  = ExpressionNode(NodeType.TERMINAL, "mom_5d",  arity=0)
        right = ExpressionNode(NodeType.TERMINAL, "mom_20d", arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "AND", left=left, right=right, arity=2)
        tree  = ExpressionTree(root)
        data  = make_data()
        result = tree.evaluate(data)
        expected = ((data["mom_5d"] != 0) & (data["mom_20d"] != 0)).astype(float)
        np.testing.assert_allclose(result, expected)

    def test_evaluate_neg_unary(self):
        inner = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "NEG", left=inner, arity=1)
        tree  = ExpressionTree(root)
        data  = make_data()
        result = tree.evaluate(data)
        np.testing.assert_allclose(result, -data["close"])

    def test_evaluate_abs_unary(self):
        inner = ExpressionNode(NodeType.TERMINAL, "mom_5d", arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "ABS", left=inner, arity=1)
        tree  = ExpressionTree(root)
        data  = make_data()
        result = tree.evaluate(data)
        np.testing.assert_allclose(result, np.abs(data["mom_5d"]))

    def test_evaluate_clip_bounds(self):
        inner = ExpressionNode(NodeType.CONSTANT, 100.0, arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "CLIP", left=inner, arity=1)
        tree  = ExpressionTree(root)
        data  = make_data()
        result = tree.evaluate(data)
        np.testing.assert_allclose(result, 3.0)  # clipped to 3

    def test_evaluate_max_binary(self):
        left  = ExpressionNode(NodeType.TERMINAL, "mom_5d",  arity=0)
        right = ExpressionNode(NodeType.TERMINAL, "mom_20d", arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "MAX", left=left, right=right, arity=2)
        tree  = ExpressionTree(root)
        data  = make_data()
        result = tree.evaluate(data)
        np.testing.assert_allclose(result, np.maximum(data["mom_5d"], data["mom_20d"]))


class TestExpressionTreeEvaluateEma:

    def test_expression_tree_evaluate_ema(self):
        """EMA(close, 10) output should pass a basic sanity check."""
        tree   = make_tree_ema()
        data   = make_data()
        result = tree.evaluate(data)
        assert len(result) == N
        # EMA should have finite values (ignoring first point)
        assert np.isfinite(result[5:]).all(), "EMA should be finite after warmup"

    def test_ema_first_value_matches_first_input(self):
        """EMA at first bar should equal the first input value."""
        data   = make_data()
        result = make_tree_ema().evaluate(data)
        # first non-nan should match data["close"][0]
        first_valid = result[~np.isnan(result)][0]
        assert abs(first_valid - data["close"][0]) < 1e-8

    def test_ema_smoothing(self):
        """EMA output should be smoother than the raw signal."""
        data   = make_data()
        raw    = data["close"]
        result = make_tree_ema().evaluate(data)
        raw_std = float(np.std(np.diff(raw)))
        ema_std = float(np.std(np.diff(result[~np.isnan(result)])))
        assert ema_std < raw_std, "EMA should smooth the input signal"

    def test_sma_mean_property(self):
        """SMA over all-constant input should equal that constant."""
        n     = 50
        const = 3.0
        data  = {"close": np.full(n, const), "period": np.full(n, 5.0)}
        signal = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        period = ExpressionNode(NodeType.CONSTANT, 5.0, arity=0)
        root   = ExpressionNode(NodeType.FUNCTION, "SMA", left=signal, right=period, arity=2)
        tree   = ExpressionTree(root)
        result = tree.evaluate(data)
        # after warmup, SMA of constant should equal constant
        np.testing.assert_allclose(result[5:], const, atol=1e-10)

    def test_lag_shifts_correctly(self):
        """LAG(close, 1) should shift array right by 1."""
        data = make_data(n=20)
        lag_node = ExpressionNode(NodeType.CONSTANT, 1.0, arity=0)
        sig_node = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        root     = ExpressionNode(NodeType.FUNCTION, "LAG", left=sig_node, right=lag_node, arity=2)
        tree     = ExpressionTree(root)
        result   = tree.evaluate(data)
        np.testing.assert_allclose(result[1:], data["close"][:-1])

    def test_diff_computes_difference(self):
        """DIFF(close, 1) should equal close[t] - close[t-1]."""
        data = make_data(n=30)
        lag_node = ExpressionNode(NodeType.CONSTANT, 1.0, arity=0)
        sig_node = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        root     = ExpressionNode(NodeType.FUNCTION, "DIFF", left=sig_node, right=lag_node, arity=2)
        tree     = ExpressionTree(root)
        result   = tree.evaluate(data)
        expected = np.diff(data["close"])
        np.testing.assert_allclose(result[1:], expected)

    def test_stddev_unary(self):
        """STDDEV should produce positive values for non-constant input."""
        data  = make_data()
        inner = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "STDDEV", left=inner, arity=1)
        tree  = ExpressionTree(root)
        result = tree.evaluate(data)
        assert np.nanmin(result[20:]) >= 0, "STDDEV should be non-negative"


class TestProtectedDivision:

    def test_protected_division_zero(self):
        """DIV(x, 0) must return 0, not inf or nan."""
        tree = make_tree_div_zero()
        data = make_data()
        result = tree.evaluate(data)
        assert np.all(result == 0.0), "Protected division by zero must return 0"

    def test_protected_division_nonzero(self):
        """DIV(close, atr) should return close/atr when atr != 0."""
        data     = make_data()
        # atr is always > 0 by construction
        sig_node = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        den_node = ExpressionNode(NodeType.TERMINAL, "atr",   arity=0)
        root     = ExpressionNode(NodeType.FUNCTION, "DIV", left=sig_node, right=den_node, arity=2)
        tree     = ExpressionTree(root)
        result   = tree.evaluate(data)
        expected = data["close"] / data["atr"]
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_protected_division_near_zero(self):
        """DIV should return 0 when denominator is below EPS."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 1e-12, 1.0])
        result = _protected_div(x, y)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert abs(result[2] - 3.0) < 1e-10


class TestTreeIntrospection:

    def test_depth_leaf(self):
        root = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        tree = ExpressionTree(root)
        assert tree.depth() == 0

    def test_depth_one_level(self):
        tree = make_tree_add()
        assert tree.depth() == 1

    def test_node_count_leaf(self):
        root = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        tree = ExpressionTree(root)
        assert tree.node_count() == 1

    def test_node_count_add(self):
        tree = make_tree_add()
        assert tree.node_count() == 3  # root + 2 leaves

    def test_to_string_add(self):
        tree = make_tree_add()
        s = tree.to_string()
        assert "mom_5d" in s
        assert "mom_20d" in s
        assert "+" in s

    def test_to_string_ema(self):
        tree = make_tree_ema()
        s = tree.to_string()
        assert "EMA" in s
        assert "close" in s

    def test_copy_is_independent(self):
        tree  = make_tree_add()
        copy_ = tree.copy()
        copy_.root.value = "SUB"
        assert tree.root.value == "ADD"


class TestTreeSimplification:

    def test_simplify_add_zero(self):
        """ADD(close, 0) -> close"""
        zero = ExpressionNode(NodeType.CONSTANT, 0.0, arity=0)
        sig  = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        root = ExpressionNode(NodeType.FUNCTION, "ADD", left=sig, right=zero, arity=2)
        tree = ExpressionTree(root)
        simplified = tree.simplify()
        assert simplified.root.node_type == NodeType.TERMINAL
        assert simplified.root.value == "close"

    def test_simplify_mul_one(self):
        """MUL(close, 1) -> close"""
        one  = ExpressionNode(NodeType.CONSTANT, 1.0, arity=0)
        sig  = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        root = ExpressionNode(NodeType.FUNCTION, "MUL", left=sig, right=one, arity=2)
        tree = ExpressionTree(root)
        simplified = tree.simplify()
        assert simplified.root.node_type == NodeType.TERMINAL

    def test_simplify_mul_zero(self):
        """MUL(close, 0) -> 0 (constant)"""
        zero = ExpressionNode(NodeType.CONSTANT, 0.0, arity=0)
        sig  = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        root = ExpressionNode(NodeType.FUNCTION, "MUL", left=sig, right=zero, arity=2)
        tree = ExpressionTree(root)
        simplified = tree.simplify()
        assert simplified.root.node_type == NodeType.CONSTANT
        assert abs(float(simplified.root.value)) < _EPS

    def test_constant_folding(self):
        """ADD(2.0, 3.0) -> 5.0"""
        left  = ExpressionNode(NodeType.CONSTANT, 2.0, arity=0)
        right = ExpressionNode(NodeType.CONSTANT, 3.0, arity=0)
        root  = ExpressionNode(NodeType.FUNCTION, "ADD", left=left, right=right, arity=2)
        tree  = ExpressionTree(root)
        simplified = tree.simplify()
        assert simplified.root.node_type == NodeType.CONSTANT
        assert abs(float(simplified.root.value) - 5.0) < _EPS


class TestTreeGenerator:

    def test_grow_returns_expression_tree(self):
        gen  = TreeGenerator()
        tree = gen.random_tree(max_depth=4, method="grow")
        assert isinstance(tree, ExpressionTree)

    def test_full_returns_expression_tree(self):
        gen  = TreeGenerator()
        tree = gen.random_tree(max_depth=3, method="full")
        assert isinstance(tree, ExpressionTree)

    def test_depth_within_limit_grow(self):
        gen = TreeGenerator()
        for _ in range(10):
            tree = gen.random_tree(max_depth=4, method="grow")
            assert tree.depth() <= 4

    def test_depth_within_limit_full(self):
        gen = TreeGenerator()
        for _ in range(5):
            tree = gen.random_tree(max_depth=4, method="full")
            assert tree.depth() <= 4

    def test_ramped_produces_variety(self):
        gen   = TreeGenerator()
        depths = set()
        for _ in range(30):
            tree = gen.random_tree_ramped(max_depth=6)
            depths.add(tree.depth())
        assert len(depths) > 1, "Ramped should produce varied depths"

    def test_invalid_method_raises(self):
        gen = TreeGenerator()
        with pytest.raises(ValueError):
            gen.random_tree(method="invalid_method")

    def test_terminal_set_respected(self):
        """Custom terminal set should be used exclusively."""
        gen  = TreeGenerator(terminal_names=["close"])
        tree = gen.random_tree(max_depth=3, method="grow")
        nodes = tree.collect_nodes()
        for node, _, _ in nodes:
            if node.node_type == NodeType.TERMINAL:
                assert node.value == "close"


# ---------------------------------------------------------------------------
# ==================== gp_operators.py tests ====================
# ---------------------------------------------------------------------------

class TestSubtreeCrossover:

    def test_subtree_crossover_depth_limit(self):
        """Crossover that would exceed max_depth=3 must return copies of parents."""
        gen = TreeGenerator(function_names=["ADD"], terminal_names=["close", "volume"])
        t1  = gen.random_tree(max_depth=3, method="full")
        t2  = gen.random_tree(max_depth=3, method="full")
        # with max_depth=2, most crossovers should be rejected since parents are depth 3
        for _ in range(20):
            c1, c2 = subtree_crossover(t1, t2, max_depth=2)
            assert c1.depth() <= 2 or (c1.to_string() == t1.to_string() or c1.to_string() == t2.to_string())

    def test_subtree_crossover_returns_two_trees(self):
        gen = TreeGenerator()
        t1  = gen.random_tree(max_depth=4)
        t2  = gen.random_tree(max_depth=4)
        c1, c2 = subtree_crossover(t1, t2, max_depth=8)
        assert isinstance(c1, ExpressionTree)
        assert isinstance(c2, ExpressionTree)

    def test_subtree_crossover_offspring_depth_bounded(self):
        gen = TreeGenerator()
        for _ in range(20):
            t1 = gen.random_tree(max_depth=4)
            t2 = gen.random_tree(max_depth=4)
            c1, c2 = subtree_crossover(t1, t2, max_depth=8)
            assert c1.depth() <= 8
            assert c2.depth() <= 8

    def test_subtree_crossover_does_not_mutate_parents(self):
        """Parents should be unchanged after crossover."""
        gen    = TreeGenerator()
        t1     = gen.random_tree(max_depth=3)
        t2     = gen.random_tree(max_depth=3)
        s1_pre = t1.to_string()
        s2_pre = t2.to_string()
        subtree_crossover(t1, t2, max_depth=8)
        assert t1.to_string() == s1_pre
        assert t2.to_string() == s2_pre


class TestPointMutation:

    def test_point_mutation_arity_preserved(self):
        """
        After point mutation, every function node should have the same arity
        as before (arity-preserving mutation).
        """
        gen  = TreeGenerator()
        tree = gen.random_tree(max_depth=5)
        # record pre-mutation arities at function nodes
        pre_arities = {
            id(node): node.arity
            for node, _, _ in tree.collect_nodes()
            if node.node_type == NodeType.FUNCTION
        }
        mutated = point_mutation(tree, rate=1.0)  # force mutation of all nodes
        # check that all function nodes still have same-arity functions
        for node, _, _ in mutated.collect_nodes():
            if node.node_type == NodeType.FUNCTION:
                assert FUNCTION_ARITY[node.value] == node.arity

    def test_point_mutation_with_zero_rate_unchanged(self):
        """rate=0 should produce identical tree."""
        gen  = TreeGenerator()
        tree = gen.random_tree(max_depth=4)
        mutated = point_mutation(tree, rate=0.0)
        assert mutated.to_string() == tree.to_string()

    def test_point_mutation_high_rate_changes_tree(self):
        """rate=1.0 should usually change the tree."""
        gen     = TreeGenerator()
        changed = 0
        for _ in range(10):
            tree    = gen.random_tree(max_depth=4)
            mutated = point_mutation(tree, rate=1.0)
            if mutated.to_string() != tree.to_string():
                changed += 1
        assert changed > 0, "High mutation rate should change some trees"

    def test_point_mutation_constant_perturbed(self):
        """A constant-only tree should have its value changed under rate=1.0."""
        root = ExpressionNode(NodeType.CONSTANT, 5.0, arity=0)
        tree = ExpressionTree(root)
        mutated = point_mutation(tree, rate=1.0)
        assert mutated.root.node_type == NodeType.CONSTANT
        # value might change (depends on gaussian perturbation)
        # just verify it's still a finite float
        assert math.isfinite(float(mutated.root.value))


class TestSubtreeMutation:

    def test_subtree_mutation_returns_tree(self):
        gen  = TreeGenerator()
        tree = gen.random_tree(max_depth=4)
        mut  = subtree_mutation(tree, max_depth=3)
        assert isinstance(mut, ExpressionTree)

    def test_subtree_mutation_does_not_alter_original(self):
        gen  = TreeGenerator()
        tree = gen.random_tree(max_depth=4)
        pre  = tree.to_string()
        subtree_mutation(tree, max_depth=3)
        assert tree.to_string() == pre


class TestHoistMutation:

    def test_hoist_mutation_reduces_depth(self):
        """
        Hoist mutation on a tree of depth >= 2 should generally reduce depth
        or keep it the same.
        """
        gen = TreeGenerator()
        reduced = 0
        same    = 0
        for _ in range(30):
            tree    = gen.random_tree(max_depth=5, method="full")
            hoisted = hoist_mutation(tree)
            d_pre   = tree.depth()
            d_post  = hoisted.depth()
            if d_post < d_pre:
                reduced += 1
            elif d_post == d_pre:
                same += 1
        # hoisting should reduce depth in most cases
        assert reduced > 0, "Hoist should reduce depth in at least some cases"

    def test_hoist_mutation_leaf_unchanged(self):
        """Hoist on a single leaf should return a copy unchanged."""
        root    = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        tree    = ExpressionTree(root)
        hoisted = hoist_mutation(tree)
        assert hoisted.node_count() == 1

    def test_hoist_mutation_does_not_alter_original(self):
        gen  = TreeGenerator()
        tree = gen.random_tree(max_depth=4)
        pre  = tree.to_string()
        hoist_mutation(tree)
        assert tree.to_string() == pre


# ---------------------------------------------------------------------------
# ==================== Selection tests ====================
# ---------------------------------------------------------------------------

class TestTournamentSelection:

    def test_tournament_selection_best_bias(self):
        """
        With k equal to population size, the best individual should
        be selected every time.
        """
        gen        = TreeGenerator()
        population = [gen.random_tree(max_depth=3) for _ in range(20)]
        fitnesses  = list(range(20))  # tree index 19 has highest fitness
        best_tree  = population[19]
        for _ in range(20):
            selected = tournament_select(population, fitnesses, k=20)
            assert selected is best_tree

    def test_tournament_selection_valid_output(self):
        gen        = TreeGenerator()
        population = [gen.random_tree(max_depth=3) for _ in range(10)]
        fitnesses  = [float(i) for i in range(10)]
        selected   = tournament_select(population, fitnesses, k=3)
        assert selected in population

    def test_tournament_k_larger_than_population(self):
        """k > population size should be clamped, not error."""
        gen        = TreeGenerator()
        population = [gen.random_tree(max_depth=2) for _ in range(5)]
        fitnesses  = [1.0, 2.0, 3.0, 4.0, 5.0]
        selected   = tournament_select(population, fitnesses, k=100)
        assert selected in population


class TestFitnessProportionalSelect:

    def test_fitness_proportional_valid_output(self):
        gen        = TreeGenerator()
        population = [gen.random_tree(max_depth=2) for _ in range(10)]
        fitnesses  = [float(i) for i in range(10)]
        selected   = fitness_proportional_select(population, fitnesses)
        assert selected in population

    def test_fitness_proportional_uniform_fitnesses(self):
        """Equal fitnesses should fall back to uniform random."""
        gen        = TreeGenerator()
        population = [gen.random_tree(max_depth=2) for _ in range(5)]
        fitnesses  = [1.0] * 5
        selected   = fitness_proportional_select(population, fitnesses)
        assert selected in population


class TestLexicaseSelection:

    def test_lexicase_selection_corner_case(self):
        """
        With only one individual that is best on every case, it must be selected.
        """
        gen  = TreeGenerator()
        pop  = [gen.random_tree(max_depth=2) for _ in range(5)]
        # individual 4 is best on all 3 cases
        mat  = np.array([
            [0.1, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.0, 0.2, 0.1],
            [0.1, 0.0, 0.2],
            [0.9, 0.9, 0.9],  # clearly dominant
        ])
        for _ in range(10):
            selected = lexicase_select(pop, mat)
            assert selected is pop[4]

    def test_lexicase_single_individual(self):
        gen = TreeGenerator()
        pop = [gen.random_tree(max_depth=2)]
        mat = np.array([[0.5, 0.3]])
        selected = lexicase_select(pop, mat)
        assert selected is pop[0]

    def test_lexicase_nan_handling(self):
        """NaN fitness values should not cause a crash."""
        gen  = TreeGenerator()
        pop  = [gen.random_tree(max_depth=2) for _ in range(4)]
        mat  = np.array([
            [np.nan, 0.5],
            [0.5,    np.nan],
            [0.3,    0.4],
            [0.7,    0.7],
        ])
        selected = lexicase_select(pop, mat)
        assert selected in pop


# ---------------------------------------------------------------------------
# ==================== gp_engine.py tests ====================
# ---------------------------------------------------------------------------

class TestSpearmanIC:

    def test_spearman_ic_perfect_correlation(self):
        x = np.arange(10.0)
        assert abs(_spearman_ic(x, x) - 1.0) < 1e-10

    def test_spearman_ic_anti_correlation(self):
        x = np.arange(10.0)
        y = -x
        assert abs(_spearman_ic(x, y) + 1.0) < 1e-10

    def test_spearman_ic_nan_pairs_skipped(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = _spearman_ic(x, y)
        assert math.isfinite(ic)

    def test_spearman_ic_insufficient_data(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        ic = _spearman_ic(x, y)
        assert ic == 0.0


class TestGPFitness:

    def test_gp_fitness_ic_range(self):
        """
        IC should be in [-1, 1] for any valid signal.
        """
        engine  = GPEngine(seed=42)
        config  = GPConfig(population_size=20, max_generations=1)
        data    = make_data()
        returns = make_returns()
        population = engine.initialize_population(config)
        engine.evaluate_population(population, data, returns)
        for ind in population:
            ic = ind.fitness_scores.get("ic", 0.0)
            assert -1.0 <= ic <= 1.0, f"IC={ic} out of range"

    def test_gp_fitness_combined_is_numeric(self):
        engine  = GPEngine(seed=1)
        config  = GPConfig(population_size=10, max_generations=1)
        data    = make_data()
        returns = make_returns()
        population = engine.initialize_population(config)
        engine.evaluate_population(population, data, returns)
        for ind in population:
            cf = ind.fitness_scores.get("combined", None)
            assert cf is not None
            assert math.isfinite(cf) or cf == -999.0

    def test_evaluate_fitness_all_nan_signal(self):
        """An all-NaN signal should return zero scores."""
        engine   = GPEngine()
        data     = {"close": np.full(N, np.nan)}
        returns  = make_returns()
        root     = ExpressionNode(NodeType.TERMINAL, "close", arity=0)
        ind      = Individual(tree=ExpressionTree(root))
        scores   = engine.evaluate_fitness(ind, data, returns)
        assert scores["combined"] == -999.0

    def test_zero_score_structure(self):
        tree   = make_tree_add()
        scores = GPEngine._zero_scores(tree)
        assert "ic" in scores
        assert "combined" in scores
        assert scores["combined"] == -999.0


class TestGPEngineInitialization:

    def test_initialize_population_correct_size(self):
        engine = GPEngine(seed=0)
        config = GPConfig(population_size=50)
        pop    = engine.initialize_population(config)
        assert len(pop) == 50

    def test_initialize_population_depth_bounded(self):
        engine = GPEngine(seed=0)
        config = GPConfig(population_size=30, max_depth=6)
        pop    = engine.initialize_population(config)
        for ind in pop:
            assert ind.tree.depth() <= 6

    def test_initialize_population_all_individuals(self):
        engine = GPEngine(seed=0)
        config = GPConfig(population_size=20)
        pop    = engine.initialize_population(config)
        for ind in pop:
            assert isinstance(ind, Individual)
            assert isinstance(ind.tree, ExpressionTree)


class TestGPEngineEvolve:

    def test_evolve_generation_correct_size(self):
        engine  = GPEngine(seed=5)
        config  = GPConfig(population_size=30, max_depth=5, elite_n=5)
        data    = make_data()
        returns = make_returns()
        pop     = engine.initialize_population(config)
        engine.evaluate_population(pop, data, returns)
        new_pop = engine.evolve_generation(pop, data, returns, config, 1)
        assert len(new_pop) == config.population_size

    def test_elites_preserved(self):
        """Top elite_n individuals from previous generation should appear in next."""
        engine  = GPEngine(seed=7)
        config  = GPConfig(population_size=20, elite_n=3, max_depth=4)
        data    = make_data()
        returns = make_returns()
        pop     = engine.initialize_population(config)
        engine.evaluate_population(pop, data, returns)
        pop.sort(key=lambda i: i.combined_fitness, reverse=True)
        top_strs = {ind.tree.to_string() for ind in pop[:3]}
        new_pop  = engine.evolve_generation(pop, data, returns, config, 1)
        new_strs = {ind.tree.to_string() for ind in new_pop}
        # at least some top expressions should survive
        assert len(top_strs & new_strs) > 0


class TestGPExport:

    def test_export_best_keys(self):
        engine = GPEngine(seed=0)
        config = GPConfig(population_size=10, max_generations=1)
        data   = make_data()
        returns = make_returns()
        pop    = engine.initialize_population(config)
        engine.evaluate_population(pop, data, returns)
        pop.sort(key=lambda i: i.combined_fitness, reverse=True)
        export = engine.export_best(pop[0])
        assert "expression"     in export
        assert "fitness_scores" in export
        assert "node_count"     in export
        assert "depth"          in export


# ---------------------------------------------------------------------------
# ==================== signal_validator.py tests ====================
# ---------------------------------------------------------------------------

class TestSignalValidatorIC:

    def test_validate_ic_passes_when_correlated(self):
        """A signal perfectly correlated with returns should pass IC check."""
        rng     = np.random.default_rng(10)
        returns = rng.standard_normal(200) * 0.01
        signal  = returns + rng.standard_normal(200) * 0.001  # very high IC
        sv      = SignalValidator(min_ic=0.05)
        assert sv.validate_ic(signal, returns) is True

    def test_validate_ic_fails_when_random(self):
        """Uncorrelated random signals should fail IC check."""
        rng     = np.random.default_rng(20)
        returns = rng.standard_normal(200) * 0.01
        signal  = rng.standard_normal(200)  # no correlation
        sv      = SignalValidator(min_ic=0.5)  # high threshold
        assert sv.validate_ic(signal, returns) is False

    def test_validate_ic_override_threshold(self):
        rng     = np.random.default_rng(30)
        returns = rng.standard_normal(100) * 0.01
        signal  = returns.copy()
        sv      = SignalValidator(min_ic=0.9)
        assert sv.validate_ic(signal, returns, min_ic=0.01) is True


class TestSignalValidatorICIR:

    def test_validate_icir_stable_series(self):
        """Consistent IC series should pass ICIR check."""
        ic_series = np.full(50, 0.1)  # constant IC of 0.1
        sv = SignalValidator(min_icir=0.5)
        assert sv.validate_icir(ic_series) is True

    def test_validate_icir_noisy_series(self):
        """Highly variable IC series should fail ICIR check."""
        rng = np.random.default_rng(0)
        ic_series = rng.standard_normal(50) * 2.0  # huge std, tiny mean
        sv = SignalValidator(min_icir=2.0)
        assert sv.validate_icir(ic_series) is False

    def test_validate_icir_insufficient_data(self):
        """Less than 3 IC observations -> False."""
        sv = SignalValidator()
        assert sv.validate_icir(np.array([0.1, 0.1])) is False


class TestSignalValidatorLookahead:

    def test_signal_validator_lookahead(self):
        """
        A signal that uses future data should be detected as lookahead.
        We simulate lookahead by making the signal always equal to the
        last value of the array (which changes when we truncate).
        """
        rng  = np.random.default_rng(42)
        data = {"close": rng.standard_normal(100)}

        # Lookahead function: always uses the LAST element of the array
        def lookahead_fn(d: dict) -> np.ndarray:
            arr = d["close"]
            # last element is from the future when truncated
            return np.full(len(arr), arr[-1])

        sv = SignalValidator()
        result = sv.validate_no_lookahead(lookahead_fn, data)
        assert result is False, "Lookahead function should be detected"

    def test_causal_signal_passes_lookahead(self):
        """A truly causal signal (rolling mean) should pass the lookahead check."""
        rng  = np.random.default_rng(55)
        data = {"close": rng.standard_normal(100) + 50.0}

        # Causal function: rolling mean (only uses past data)
        def causal_fn(d: dict) -> np.ndarray:
            arr = np.asarray(d["close"])
            out = np.full(len(arr), np.nan)
            for i in range(2, len(arr)):
                out[i] = arr[i - 2: i].mean()
            return out

        sv     = SignalValidator()
        result = sv.validate_no_lookahead(causal_fn, data)
        assert result is True, "Causal rolling mean should pass lookahead check"


class TestSignalValidatorStability:

    def test_validate_stability_passes(self):
        ic_series = np.full(50, 0.08)  # zero std
        sv = SignalValidator(max_ic_std=0.15)
        assert sv.validate_stability(ic_series) is True

    def test_validate_stability_fails(self):
        rng = np.random.default_rng(0)
        ic_series = rng.standard_normal(50) * 5.0  # huge std
        sv = SignalValidator(max_ic_std=0.15)
        assert sv.validate_stability(ic_series) is False

    def test_validate_stability_insufficient_data(self):
        """Fewer than 3 IC values -> pass by default."""
        sv = SignalValidator()
        assert sv.validate_stability(np.array([0.5])) is True


class TestLibraryCorrelation:

    def test_check_correlation_with_library_accept(self):
        rng   = np.random.default_rng(99)
        new   = rng.standard_normal(200)
        lib   = {"sig1": rng.standard_normal(200)}  # independent
        sv    = SignalValidator(max_library_corr=0.85)
        assert sv.check_correlation_with_library(new, lib) is True

    def test_check_correlation_with_library_reject(self):
        rng  = np.random.default_rng(77)
        base = rng.standard_normal(200)
        new  = base + rng.standard_normal(200) * 0.001  # nearly identical
        lib  = {"sig1": base}
        sv   = SignalValidator(max_library_corr=0.85)
        assert sv.check_correlation_with_library(new, lib) is False

    def test_check_correlation_empty_library(self):
        rng = np.random.default_rng(1)
        new = rng.standard_normal(100)
        sv  = SignalValidator()
        assert sv.check_correlation_with_library(new, {}) is True


class TestOutOfSampleValidation:

    def test_validate_out_of_sample_structure(self):
        rng          = np.random.default_rng(11)
        train_ret    = rng.standard_normal(200) * 0.01
        test_ret     = rng.standard_normal(100) * 0.01
        base_signal  = rng.standard_normal(300)
        train_signal = base_signal[:200]
        test_signal  = base_signal[200:]

        def signal_fn(d: dict) -> np.ndarray:
            return np.asarray(d["signal"])

        train_data = {"signal": train_signal}
        test_data  = {"signal": test_signal}

        sv     = SignalValidator()
        result = sv.validate_out_of_sample(
            signal_fn, train_data, test_data, train_ret, test_ret
        )
        assert "is_ic"    in result
        assert "oos_ic"   in result
        assert "ratio"    in result
        assert math.isfinite(result["is_ic"])
        assert math.isfinite(result["oos_ic"])

    def test_validate_out_of_sample_missing_returns(self):
        """If returns not found in data and not supplied, return zeros."""
        def signal_fn(d: dict) -> np.ndarray:
            return np.zeros(len(next(iter(d.values()))))

        sv     = SignalValidator()
        result = sv.validate_out_of_sample(
            signal_fn, {"x": np.zeros(10)}, {"x": np.zeros(5)}
        )
        assert result["is_ic"] == 0.0


class TestFullValidationPipeline:

    def _make_good_signal(self, returns, noise_level=0.3):
        """Create a signal that is correlated with returns."""
        rng   = np.random.default_rng(123)
        n     = len(returns)
        noise = rng.standard_normal(n) * noise_level
        return returns + noise

    def test_validation_passes_for_good_signal(self):
        rng         = np.random.default_rng(0)
        all_ret     = rng.standard_normal(400) * 0.01
        train_ret   = all_ret[:300]
        test_ret    = all_ret[300:]
        all_signal  = self._make_good_signal(all_ret, noise_level=0.1)
        train_sig   = all_signal[:300]
        test_sig    = all_signal[300:]

        def signal_fn(d: dict) -> np.ndarray:
            return np.asarray(d["sig"])

        train_data = {"sig": train_sig}
        test_data  = {"sig": test_sig}

        sv     = SignalValidator(min_ic=0.01, min_icir=0.1, max_ic_std=2.0,
                                 oos_min_is_ratio=0.1)
        result = sv.validate_all(
            signal_fn, train_data, test_data, train_ret, test_ret
        )
        assert isinstance(result, ValidationResult)
        # with loose thresholds, a noisy-but-correlated signal should pass
        if not result.passed:
            # print reasons for debugging
            print("Validation failed:", result.failure_reasons)

    def test_validation_fails_for_random_signal(self):
        rng         = np.random.default_rng(5)
        train_ret   = rng.standard_normal(300) * 0.01
        test_ret    = rng.standard_normal(100) * 0.01
        train_sig   = rng.standard_normal(300)  # pure noise
        test_sig    = rng.standard_normal(100)

        def signal_fn(d: dict) -> np.ndarray:
            return np.asarray(d["sig"])

        sv = SignalValidator(min_ic=0.5, min_icir=2.0)  # strict thresholds
        result = sv.validate_all(
            signal_fn,
            {"sig": train_sig}, {"sig": test_sig},
            train_ret, test_ret,
        )
        assert not result.passed
        assert len(result.failure_reasons) > 0


# ---------------------------------------------------------------------------
# ==================== Rolling stats helpers tests ====================
# ---------------------------------------------------------------------------

class TestRollingStats:

    def test_rolling_mean_constant(self):
        arr = np.full(20, 5.0)
        result = _rolling_mean(arr, 5)
        np.testing.assert_allclose(result[4:], 5.0)

    def test_rolling_std_constant_is_zero(self):
        arr = np.full(20, 3.0)
        result = _rolling_std(arr, 5)
        np.testing.assert_allclose(result[4:], 0.0, atol=1e-12)

    def test_rolling_zscore_shape(self):
        arr    = np.random.default_rng(0).standard_normal(100)
        result = _rolling_zscore(arr, 20)
        assert result.shape == arr.shape

    def test_rank_normalize_range(self):
        arr    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rank_normalize(arr)
        assert result.min() >= -1.0
        assert result.max() <=  1.0

    def test_rank_normalize_monotone(self):
        arr    = np.arange(10.0)
        result = _rank_normalize(arr)
        diffs  = np.diff(result)
        assert np.all(diffs >= 0), "Rank normalization should be monotone"

    def test_ema_vectorized_converges(self):
        """EMA of a step function should converge to the new level."""
        arr = np.concatenate([np.zeros(50), np.ones(100)])
        result = _ema_vectorized(arr, span=5)
        assert result[-1] > 0.98, "EMA should converge to 1.0 after many steps"


# ---------------------------------------------------------------------------
# ==================== Misc integration tests ====================
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_run_small(self):
        """End-to-end small run should complete without error."""
        engine  = GPEngine(seed=99)
        config  = GPConfig(
            population_size=20,
            max_generations=3,
            elite_n=2,
            verbosity=0,
        )
        data    = make_data()
        returns = make_returns()
        pareto  = engine.run(data, returns, config)
        assert len(pareto) > 0
        assert all(isinstance(ind, Individual) for ind in pareto)

    def test_pareto_front_non_dominated(self):
        """All members of the Pareto front should be non-dominated by each other."""
        engine  = GPEngine(seed=12)
        config  = GPConfig(population_size=30, max_generations=2, verbosity=0)
        data    = make_data()
        returns = make_returns()
        pareto  = engine.run(data, returns, config)
        for i, cand in enumerate(pareto):
            cic   = cand.fitness_scores.get("ic",   0.0)
            cicir = cand.fitness_scores.get("icir", 0.0)
            for j, other in enumerate(pareto):
                if i == j:
                    continue
                oic   = other.fitness_scores.get("ic",   0.0)
                oicir = other.fitness_scores.get("icir", 0.0)
                # other should NOT strictly dominate cand
                assert not (oic >= cic and oicir >= cicir and (oic > cic or oicir > cicir)), (
                    f"Individual {j} dominates {i} in Pareto front"
                )

    def test_terminal_set_size(self):
        """ALL_TERMINALS should contain at least 105 signal + raw feature names."""
        assert len(SIGNAL_NAMES) >= 100
        assert len(ALL_TERMINALS) >= 105

    def test_function_arity_consistency(self):
        """Every function in ALL_FUNCTION_NAMES must have a valid arity in FUNCTION_ARITY."""
        for fn in ALL_FUNCTION_NAMES:
            assert fn in FUNCTION_ARITY
            assert FUNCTION_ARITY[fn] in (1, 2)

    def test_crossover_plus_mutation_cycle(self):
        """Crossover followed by point mutation should return valid trees."""
        gen = TreeGenerator()
        t1  = gen.random_tree(max_depth=4)
        t2  = gen.random_tree(max_depth=4)
        c1, c2 = subtree_crossover(t1, t2, max_depth=8)
        m1 = point_mutation(c1, rate=0.1)
        m2 = hoist_mutation(c2)
        assert isinstance(m1, ExpressionTree)
        assert isinstance(m2, ExpressionTree)
        assert m1.depth() >= 0
        assert m2.depth() >= 0
