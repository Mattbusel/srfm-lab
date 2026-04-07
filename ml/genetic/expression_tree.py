"""
expression_tree.py -- Symbolic expression trees for GP-based signal discovery.

Provides node types, function/terminal sets, tree evaluation (vectorized on
numpy arrays), tree generation (grow / full / ramped half-and-half), and
algebraic simplification with constant folding.

No external dependencies beyond numpy.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Node and function enumerations
# ---------------------------------------------------------------------------

class NodeType(Enum):
    TERMINAL = auto()   # leaf: variable name or raw feature
    FUNCTION = auto()   # internal: operator / financial function
    CONSTANT = auto()   # leaf: float literal


class FunctionArity(Enum):
    UNARY = 1
    BINARY = 2


# ---------------------------------------------------------------------------
# All supported functions with their arities
# ---------------------------------------------------------------------------

# (name, arity)
_BINARY_FUNCTIONS: List[Tuple[str, int]] = [
    ("ADD", 2),
    ("SUB", 2),
    ("MUL", 2),
    ("DIV", 2),   # protected
    ("GT",  2),
    ("LT",  2),
    ("EQ",  2),
    ("AND", 2),
    ("OR",  2),
    ("MAX", 2),
    ("MIN", 2),
    ("EMA", 2),   # EMA(signal, period_constant)
    ("SMA", 2),   # SMA(signal, period_constant)
    ("LAG", 2),   # LAG(signal, int_constant)
    ("DIFF", 2),  # DIFF(signal, int_constant)
    ("CORR", 2),  # rolling correlation of two signals (window=20)
]

_UNARY_FUNCTIONS: List[Tuple[str, int]] = [
    ("NEG",    1),
    ("ABS",    1),
    ("SIGN",   1),
    ("NOT",    1),
    ("STDDEV", 1),   # rolling stddev, window=20
    ("RANK",   1),   # cross-sectional rank normalization
    ("ZSCORE", 1),   # rolling z-score, window=20
    ("LOG",    1),   # log(|x| + 1e-8)
    ("SQRT",   1),   # sqrt(|x|)
    ("CLIP",   1),   # clip to [-3, 3]
]

_ALL_FUNCTIONS: List[Tuple[str, int]] = _BINARY_FUNCTIONS + _UNARY_FUNCTIONS

FUNCTION_ARITY: Dict[str, int] = {name: arity for name, arity in _ALL_FUNCTIONS}

BINARY_FUNCTION_NAMES: List[str] = [name for name, _ in _BINARY_FUNCTIONS]
UNARY_FUNCTION_NAMES:  List[str] = [name for name, _ in _UNARY_FUNCTIONS]
ALL_FUNCTION_NAMES:    List[str] = [name for name, _ in _ALL_FUNCTIONS]


# ---------------------------------------------------------------------------
# Terminal set -- 105 signals from research/signal_analytics/signal_library.py
# plus raw features
# ---------------------------------------------------------------------------

RAW_FEATURES: List[str] = [
    "close", "volume", "atr", "bh_mass", "hurst_h", "nav_omega",
    "high", "low", "open",
]

SIGNAL_NAMES: List[str] = [
    # MOMENTUM (20)
    "mom_1d", "mom_5d", "mom_20d", "mom_60d", "mom_252d",
    "mom_sharpe", "mom_acceleration", "mom_52w_high", "mom_crash_protection",
    "mom_ts_moskowitz", "mom_cs_rank", "mom_seasonality", "mom_dual",
    "mom_absolute", "mom_intermediate", "mom_short_reversal", "mom_end_of_month",
    "mom_gap", "mom_volume_weighted", "mom_up_down_volume",
    # extra momentum
    "mom_tick_proxy", "mom_price_accel", "mom_multi_tf_composite",
    # MEAN REVERSION (20)
    "mr_zscore_10", "mr_zscore_20", "mr_zscore_50",
    "mr_bollinger_position", "mr_rsi", "mr_linreg_residual",
    "mr_sma_deviation", "mr_kalman_residual", "mr_pairs_ratio_zscore",
    "mr_ou_weighted", "mr_vwap_deviation", "mr_price_oscillator",
    "mr_dpo", "mr_cci", "mr_williams_r", "mr_stochastic_k",
    "mr_chande_momentum", "mr_roc_mean_rev", "mr_price_channel",
    "mr_log_autoregression", "mr_hurst_adjusted",
    # VOLATILITY (15)
    "vol_ewma_forecast", "vol_realized_5d", "vol_realized_20d",
    "vol_realized_60d", "vol_of_vol", "vol_regime", "vol_atr_percentile",
    "vol_skew_proxy", "vol_term_structure", "vol_parkinson",
    "vol_garman_klass", "vol_yang_zhang", "vol_rogers_satchell",
    "vol_arch_signal", "vol_normalized_range",
    # (removing vol_hist_vs_implied -- needs implied vol data)
    # MICROSTRUCTURE (15)
    "ms_volume_surprise", "ms_vpt", "ms_obv_normalized",
    "ms_cmf", "ms_adl", "ms_force_index", "ms_emv",
    "ms_volume_oscillator", "ms_mfi", "ms_nvi", "ms_pvi",
    "ms_pvt", "ms_volume_momentum", "ms_large_trade", "ms_kyle_lambda",
    # PHYSICS / SRFM (15)
    "phys_bh_mass", "phys_proper_time", "phys_timelike_fraction",
    "phys_ds2_trend", "phys_bh_formation_rate", "phys_geodesic_deviation",
    "phys_angular_velocity", "phys_hurst_signal", "phys_fractal_dimension",
    "phys_hawking_temperature", "phys_grav_lensing", "phys_phase_transition",
    "phys_causal_info_ratio", "phys_regime_velocity", "phys_curvature_proxy",
    # TECHNICAL (15)
    "tech_macd_histogram", "tech_adx", "tech_aroon_oscillator",
    "tech_cci", "tech_keltner_position", "tech_donchian_breakout",
    "tech_ichimoku_cloud", "tech_psar", "tech_elder_ray",
    "tech_vortex", "tech_chande_kroll", "tech_supertrend",
    "tech_trix", "tech_mass_index", "tech_ulcer_index",
]

ALL_TERMINALS: List[str] = RAW_FEATURES + SIGNAL_NAMES


# ---------------------------------------------------------------------------
# ExpressionNode dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExpressionNode:
    """A single node in the symbolic expression tree."""
    node_type: NodeType
    value: Any              # str (function/terminal name) or float (constant)
    left:  Optional["ExpressionNode"] = field(default=None, repr=False)
    right: Optional["ExpressionNode"] = field(default=None, repr=False)
    arity: int = 0          # 0 for terminals/constants, 1 or 2 for functions

    def is_leaf(self) -> bool:
        return self.node_type in (NodeType.TERMINAL, NodeType.CONSTANT)

    def copy(self) -> "ExpressionNode":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Vectorized evaluation helpers
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW = 20
_EPS = 1e-8


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using cumsum trick."""
    if window <= 1:
        return arr.copy()
    n = len(arr)
    out = np.full(n, np.nan)
    cs = np.nancumsum(arr)
    out[window - 1:] = (cs[window - 1:] - np.concatenate([[0], cs[: n - window]])) / window
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (ddof=1)."""
    if window <= 1:
        return np.zeros(len(arr))
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = arr[i - window + 1: i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= 2:
            out[i] = float(np.std(valid, ddof=1))
        elif len(valid) == 1:
            out[i] = 0.0
    return out


def _ema_vectorized(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with span (alpha = 2/(span+1))."""
    if span <= 0:
        span = 1
    alpha = 2.0 / (span + 1)
    n = len(arr)
    out = np.full(n, np.nan)
    # find first non-nan
    start = 0
    while start < n and np.isnan(arr[start]):
        start += 1
    if start >= n:
        return out
    out[start] = arr[start]
    for i in range(start + 1, n):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _sma_vectorized(arr: np.ndarray, window: int) -> np.ndarray:
    return _rolling_mean(arr, max(1, int(window)))


def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Map values to [-1, 1] via rank normalization, ignoring NaN."""
    n = len(arr)
    out = np.full(n, np.nan)
    valid_mask = ~np.isnan(arr)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return out
    vals = arr[valid_idx]
    ranks = np.argsort(np.argsort(vals)).astype(float)
    if len(ranks) > 1:
        ranks = ranks / (len(ranks) - 1) * 2.0 - 1.0
    else:
        ranks = np.zeros(1)
    out[valid_idx] = ranks
    return out


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score: (x - mean) / std over window."""
    means = _rolling_mean(arr, window)
    stds  = _rolling_std(arr, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(stds > _EPS, (arr - means) / stds, 0.0)
    return out


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling Pearson correlation."""
    n = len(a)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        sa = a[i - window + 1: i + 1]
        sb = b[i - window + 1: i + 1]
        mask = ~(np.isnan(sa) | np.isnan(sb))
        if mask.sum() < 4:
            continue
        x, y = sa[mask], sb[mask]
        std_x = np.std(x, ddof=1)
        std_y = np.std(y, ddof=1)
        if std_x < _EPS or std_y < _EPS:
            out[i] = 0.0
        else:
            out[i] = float(np.corrcoef(x, y)[0, 1])
    return out


def _protected_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Protected division: returns 0 where |y| <= EPS."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(y) > _EPS, x / y, 0.0)
    return result


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.abs(x) + _EPS)


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x))


# ---------------------------------------------------------------------------
# ExpressionTree
# ---------------------------------------------------------------------------

class ExpressionTree:
    """
    Symbolic expression tree that evaluates to a numpy array signal.

    Evaluation is fully vectorized -- each node returns an ndarray of the
    same length as the input data arrays.
    """

    def __init__(self, root: ExpressionNode):
        self.root = root

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Evaluate the expression tree on a data dictionary.

        data maps terminal names (signal names or raw feature names) to
        1-D numpy arrays of equal length.

        Returns a 1-D numpy array of float64.
        """
        if not data:
            raise ValueError("data dict is empty")
        # infer length from first array
        n = len(next(iter(data.values())))
        return self._eval_node(self.root, data, n)

    def _eval_node(
        self, node: ExpressionNode, data: Dict[str, np.ndarray], n: int
    ) -> np.ndarray:
        if node.node_type == NodeType.CONSTANT:
            return np.full(n, float(node.value))

        if node.node_type == NodeType.TERMINAL:
            name = node.value
            if name in data:
                arr = np.asarray(data[name], dtype=np.float64)
                if len(arr) != n:
                    raise ValueError(
                        f"Terminal '{name}' length {len(arr)} != expected {n}"
                    )
                return arr
            # terminal not in data -- return zeros with NaN prefix to signal absence
            return np.full(n, 0.0)

        # FUNCTION node
        fn = node.value
        arity = node.arity

        if arity == 1:
            left_val = self._eval_node(node.left, data, n)
            return self._apply_unary(fn, left_val)
        else:
            left_val  = self._eval_node(node.left, data, n)
            right_val = self._eval_node(node.right, data, n)
            return self._apply_binary(fn, left_val, right_val)

    @staticmethod
    def _apply_unary(fn: str, x: np.ndarray) -> np.ndarray:
        if fn == "NEG":
            return -x
        if fn == "ABS":
            return np.abs(x)
        if fn == "SIGN":
            return np.sign(x)
        if fn == "NOT":
            return (x == 0.0).astype(np.float64)
        if fn == "STDDEV":
            return _rolling_std(x, _DEFAULT_WINDOW)
        if fn == "RANK":
            return _rank_normalize(x)
        if fn == "ZSCORE":
            return _rolling_zscore(x, _DEFAULT_WINDOW)
        if fn == "LOG":
            return _safe_log(x)
        if fn == "SQRT":
            return _safe_sqrt(x)
        if fn == "CLIP":
            return np.clip(x, -3.0, 3.0)
        raise ValueError(f"Unknown unary function: {fn}")

    @staticmethod
    def _apply_binary(fn: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if fn == "ADD":
            return x + y
        if fn == "SUB":
            return x - y
        if fn == "MUL":
            return x * y
        if fn == "DIV":
            return _protected_div(x, y)
        if fn == "GT":
            return (x > y).astype(np.float64)
        if fn == "LT":
            return (x < y).astype(np.float64)
        if fn == "EQ":
            return (np.abs(x - y) < _EPS).astype(np.float64)
        if fn == "AND":
            return ((x != 0.0) & (y != 0.0)).astype(np.float64)
        if fn == "OR":
            return ((x != 0.0) | (y != 0.0)).astype(np.float64)
        if fn == "MAX":
            return np.maximum(x, y)
        if fn == "MIN":
            return np.minimum(x, y)
        if fn == "EMA":
            # y is (nominally) a period array; use median value as span
            span = max(1, int(np.nanmedian(y)))
            return _ema_vectorized(x, span)
        if fn == "SMA":
            window = max(1, int(np.nanmedian(y)))
            return _sma_vectorized(x, window)
        if fn == "LAG":
            lag = max(1, int(np.nanmedian(y)))
            result = np.full(len(x), np.nan)
            result[lag:] = x[:-lag]
            return result
        if fn == "DIFF":
            lag = max(1, int(np.nanmedian(y)))
            result = np.full(len(x), np.nan)
            result[lag:] = x[lag:] - x[:-lag]
            return result
        if fn == "CORR":
            return _rolling_corr(x, y, window=_DEFAULT_WINDOW)
        raise ValueError(f"Unknown binary function: {fn}")

    # ------------------------------------------------------------------
    # Tree introspection
    # ------------------------------------------------------------------

    def depth(self) -> int:
        """Maximum depth of the tree (root has depth 0)."""
        return self._depth_node(self.root)

    @staticmethod
    def _depth_node(node: Optional[ExpressionNode]) -> int:
        if node is None or node.is_leaf():
            return 0
        left_d  = ExpressionTree._depth_node(node.left)
        right_d = ExpressionTree._depth_node(node.right)
        return 1 + max(left_d, right_d)

    def node_count(self) -> int:
        """Total number of nodes in the tree."""
        return self._count_node(self.root)

    @staticmethod
    def _count_node(node: Optional[ExpressionNode]) -> int:
        if node is None:
            return 0
        return (1
                + ExpressionTree._count_node(node.left)
                + ExpressionTree._count_node(node.right))

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def to_string(self) -> str:
        """Human-readable infix notation."""
        return self._node_to_str(self.root)

    @staticmethod
    def _node_to_str(node: Optional[ExpressionNode]) -> str:
        if node is None:
            return ""
        if node.node_type == NodeType.CONSTANT:
            v = node.value
            return f"{v:.4g}" if isinstance(v, float) else str(v)
        if node.node_type == NodeType.TERMINAL:
            return str(node.value)
        # function
        fn = node.value
        if node.arity == 1:
            inner = ExpressionTree._node_to_str(node.left)
            return f"{fn}({inner})"
        # binary
        infix_ops = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/",
                     "GT": ">", "LT": "<", "EQ": "==",
                     "AND": "&&", "OR": "||"}
        if fn in infix_ops:
            lstr = ExpressionTree._node_to_str(node.left)
            rstr = ExpressionTree._node_to_str(node.right)
            return f"({lstr} {infix_ops[fn]} {rstr})"
        # functional form for EMA, SMA, LAG, DIFF, CORR, MAX, MIN
        lstr = ExpressionTree._node_to_str(node.left)
        rstr = ExpressionTree._node_to_str(node.right)
        return f"{fn}({lstr}, {rstr})"

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"ExpressionTree(depth={self.depth()}, nodes={self.node_count()}, expr={self.to_string()[:60]})"

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "ExpressionTree":
        """Deep copy."""
        return ExpressionTree(copy.deepcopy(self.root))

    # ------------------------------------------------------------------
    # Simplification
    # ------------------------------------------------------------------

    def simplify(self) -> "ExpressionTree":
        """
        Algebraic simplification with constant folding.
        -- ADD(x, 0) -> x
        -- MUL(x, 1) -> x
        -- MUL(x, 0) -> 0
        -- DIV(x, 1) -> x
        -- SUB(x, 0) -> x
        -- Constant folding: any function applied only to constants becomes a constant
        """
        new_root = self._simplify_node(copy.deepcopy(self.root))
        return ExpressionTree(new_root)

    def _simplify_node(self, node: ExpressionNode) -> ExpressionNode:
        if node is None or node.is_leaf():
            return node

        # recurse first
        if node.left is not None:
            node.left = self._simplify_node(node.left)
        if node.right is not None:
            node.right = self._simplify_node(node.right)

        fn = node.value

        # constant folding: both children are constants
        if node.arity == 2:
            if (node.left is not None and node.left.node_type == NodeType.CONSTANT
                    and node.right is not None and node.right.node_type == NodeType.CONSTANT):
                lv = float(node.left.value)
                rv = float(node.right.value)
                result = self._fold_binary(fn, lv, rv)
                if result is not None:
                    return ExpressionNode(NodeType.CONSTANT, result)

        if node.arity == 1:
            if node.left is not None and node.left.node_type == NodeType.CONSTANT:
                lv = float(node.left.value)
                result = self._fold_unary(fn, lv)
                if result is not None:
                    return ExpressionNode(NodeType.CONSTANT, result)

        # algebraic identities
        if fn == "ADD":
            if self._is_const(node.right, 0.0):
                return node.left
            if self._is_const(node.left, 0.0):
                return node.right
        elif fn == "SUB":
            if self._is_const(node.right, 0.0):
                return node.left
        elif fn == "MUL":
            if self._is_const(node.right, 1.0):
                return node.left
            if self._is_const(node.left, 1.0):
                return node.right
            if self._is_const(node.right, 0.0) or self._is_const(node.left, 0.0):
                return ExpressionNode(NodeType.CONSTANT, 0.0)
        elif fn == "DIV":
            if self._is_const(node.right, 1.0):
                return node.left
        elif fn == "NEG":
            # NEG(NEG(x)) -> x
            if (node.left is not None
                    and node.left.node_type == NodeType.FUNCTION
                    and node.left.value == "NEG"):
                return node.left.left

        return node

    @staticmethod
    def _is_const(node: Optional[ExpressionNode], val: float) -> bool:
        return (node is not None
                and node.node_type == NodeType.CONSTANT
                and abs(float(node.value) - val) < _EPS)

    @staticmethod
    def _fold_binary(fn: str, lv: float, rv: float) -> Optional[float]:
        try:
            if fn == "ADD":  return lv + rv
            if fn == "SUB":  return lv - rv
            if fn == "MUL":  return lv * rv
            if fn == "DIV":  return lv / rv if abs(rv) > _EPS else 0.0
            if fn == "GT":   return float(lv > rv)
            if fn == "LT":   return float(lv < rv)
            if fn == "EQ":   return float(abs(lv - rv) < _EPS)
            if fn == "AND":  return float((lv != 0.0) and (rv != 0.0))
            if fn == "OR":   return float((lv != 0.0) or  (rv != 0.0))
            if fn == "MAX":  return max(lv, rv)
            if fn == "MIN":  return min(lv, rv)
        except Exception:
            pass
        return None

    @staticmethod
    def _fold_unary(fn: str, lv: float) -> Optional[float]:
        try:
            if fn == "NEG":    return -lv
            if fn == "ABS":    return abs(lv)
            if fn == "SIGN":   return math.copysign(1.0, lv) if lv != 0 else 0.0
            if fn == "NOT":    return float(lv == 0.0)
            if fn == "LOG":    return math.log(abs(lv) + _EPS)
            if fn == "SQRT":   return math.sqrt(abs(lv))
            if fn == "CLIP":   return max(-3.0, min(3.0, lv))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Node collection utilities (used by operators)
    # ------------------------------------------------------------------

    def collect_nodes(self) -> List[Tuple[ExpressionNode, Optional[ExpressionNode], str]]:
        """
        Return list of (node, parent, side) tuples for all nodes.
        side is 'left', 'right', or 'root'.
        """
        result: List[Tuple[ExpressionNode, Optional[ExpressionNode], str]] = []
        self._collect(self.root, None, "root", result)
        return result

    @staticmethod
    def _collect(
        node: Optional[ExpressionNode],
        parent: Optional[ExpressionNode],
        side: str,
        result: list,
    ) -> None:
        if node is None:
            return
        result.append((node, parent, side))
        ExpressionTree._collect(node.left,  node, "left",  result)
        ExpressionTree._collect(node.right, node, "right", result)

    def get_subtree_root_at_random(
        self,
    ) -> Tuple[ExpressionNode, Optional[ExpressionNode], str]:
        """Pick a uniformly random node and return (node, parent, side)."""
        nodes = self.collect_nodes()
        return random.choice(nodes)

    def set_subtree(
        self,
        parent: Optional[ExpressionNode],
        side: str,
        new_subtree: ExpressionNode,
    ) -> None:
        """Replace the child of parent on the given side with new_subtree."""
        if side == "root":
            self.root = new_subtree
        elif side == "left" and parent is not None:
            parent.left = new_subtree
        elif side == "right" and parent is not None:
            parent.right = new_subtree


# ---------------------------------------------------------------------------
# TreeGenerator
# ---------------------------------------------------------------------------

class TreeGenerator:
    """
    Generates random expression trees using grow or full method.
    Uses the combined function set and terminal set.
    """

    def __init__(
        self,
        function_names: Optional[List[str]] = None,
        terminal_names: Optional[List[str]] = None,
        const_range: Tuple[float, float] = (-5.0, 5.0),
        const_prob: float = 0.15,
    ):
        self.function_names = function_names or ALL_FUNCTION_NAMES
        self.terminal_names = terminal_names or ALL_TERMINALS
        self.const_range = const_range
        self.const_prob = const_prob

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _make_terminal_node(self) -> ExpressionNode:
        """Return either a constant node or a terminal node."""
        if random.random() < self.const_prob:
            val = random.uniform(*self.const_range)
            return ExpressionNode(NodeType.CONSTANT, val, arity=0)
        name = random.choice(self.terminal_names)
        return ExpressionNode(NodeType.TERMINAL, name, arity=0)

    def _make_function_node(self, fn_name: str) -> ExpressionNode:
        arity = FUNCTION_ARITY[fn_name]
        return ExpressionNode(NodeType.FUNCTION, fn_name, arity=arity)

    def _grow(self, max_depth: int, current_depth: int) -> ExpressionNode:
        """
        Grow method: at each interior position, randomly choose terminal or function.
        Stops at max_depth with a forced terminal.
        """
        if current_depth >= max_depth:
            return self._make_terminal_node()

        # probability of choosing a function decreases toward max_depth
        fn_prob = 0.7 if current_depth < max_depth - 1 else 0.3
        if random.random() < fn_prob:
            fn_name = random.choice(self.function_names)
            node = self._make_function_node(fn_name)
            node.left = self._grow(max_depth, current_depth + 1)
            if node.arity == 2:
                node.right = self._grow(max_depth, current_depth + 1)
            return node
        return self._make_terminal_node()

    def _full(self, max_depth: int, current_depth: int) -> ExpressionNode:
        """
        Full method: interior nodes are always functions until max_depth,
        where all leaves are terminals.
        """
        if current_depth >= max_depth:
            return self._make_terminal_node()
        fn_name = random.choice(self.function_names)
        node = self._make_function_node(fn_name)
        node.left = self._full(max_depth, current_depth + 1)
        if node.arity == 2:
            node.right = self._full(max_depth, current_depth + 1)
        return node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def random_tree(
        self, max_depth: int = 4, method: str = "grow"
    ) -> ExpressionTree:
        """
        Generate a random tree using 'grow' or 'full' method.
        Recursion starts at depth 0.
        """
        if method == "grow":
            root = self._grow(max_depth, 0)
        elif method == "full":
            root = self._full(max_depth, 0)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'grow' or 'full'.")
        return ExpressionTree(root)

    def random_tree_ramped(self, max_depth: int = 6) -> ExpressionTree:
        """
        Ramped half-and-half: evenly distribute depths from 2..max_depth,
        alternating grow / full.
        """
        depth = random.randint(2, max_depth)
        method = "grow" if random.random() < 0.5 else "full"
        return self.random_tree(max_depth=depth, method=method)
